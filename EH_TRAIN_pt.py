import time
from random import shuffle

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from EH_UTILS_pt import *
from RWNet_pt import RWNet

EXP_DATA = 'HEVC_RWNet_pt_QP37_20211129'  # 模型存放文件夹
LOW_DATA_PATH = r"D:\tian\HM-network\trainSet\HEVC_rec\qp37"  # 数据存放路径
HIGH_DATA_PATH = r"D:\tian\HM-network\trainSet\HEVC_gt"  # 标签存放路径
LOG_PATH = "./logs/%s/" % EXP_DATA  # log路径
MODEL_PATH = "./checkpoints/%s/" % EXP_DATA  # 存储训练好的模型
PATCH_SIZE = (64, 64)  # 卷积神经网络中输入图像的大小
PRE_BATCH_SIZE = 6400  # 加载数据集的BATCH数
BATCH_SIZE = 64  # 神经网络的BATCH数
BASE_LR = 1e-3  # 基础学习率
MAX_EPOCH = 800  # 训练轮次
LR_ATTENUATION_EPOCHS = 20  # 学习率衰减轮次
BATCH_SIZE_ATTENUATION_EPOCHS = 200  # batch_size衰减轮次
RESUME = True  # 是否断点续训
LAST_CHECKPOINT = "./checkpoints/HEVC_RWNet_pt_QP37_20211129/HEVC_RWNet_pt_QP37_20211129_385.pth"  # 断点路径

if __name__ == '__main__':
    # 载入数据集
    train_list = get_train_list(load_file_list(LOW_DATA_PATH), load_file_list(HIGH_DATA_PATH))

    # 如果log和model文件夹不存在，则创建
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))

    # 导入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RWNet().to(device).train()
    # # 初始化正则化
    # if weight_decay > 0:
    #     reg_loss = Regularization(model, weight_decay, p=0).to(device)
    # else:
    #     print("no regularization")
    # 开始神经网络的计算,但是这里我们使用优化器帮我们更新参数
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-2)
    # 即每个scheduler.step都衰减lr = lr * gamma,即进行指数衰减
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    # 断点续训， 从最后一个模型中加载各种参数
    if RESUME:
        path_checkpoint = LAST_CHECKPOINT
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
    # 记录每一轮次的loss和lr
    log = os.path.join(LOG_PATH, "0000")
    toatal_tb = SummaryWriter(log)

    last_epoch = start_epoch
    now_batch_size = BATCH_SIZE
    for epoch in range(last_epoch, last_epoch + MAX_EPOCH):
        # 模型、日志保存路径
        model_file = os.path.join(MODEL_PATH, "%s_%03d.pth" % (EXP_DATA, epoch))
        log_file = os.path.join(LOG_PATH, "%03d" % epoch)
        tb = SummaryWriter(log_file)
        epoch_time = time.time()
        temp_loss = 1000000
        log_loss = 0
        for idx in range(10):
            total_get_data_time, total_load_data_time, total_network_time = 0, 0, 0
            get_data_time = time.time()
            # 数据集处理
            shuffle(train_list)
            input_data, gt_data = prepare_nn_data(train_list)
            total_get_data_time += (time.time() - get_data_time)
            load_data_time = time.time()
            # 制作网络输入数据集
            input_tensor_data, gt_tensor_data = torch.from_numpy(input_data).to(device), torch.from_numpy(gt_data).to(device)
            torch_dataset = torch.utils.data.TensorDataset(input_tensor_data, gt_tensor_data)  # 得到一个元组(x, y)
            loader = torch.utils.data.DataLoader(
                dataset=torch_dataset,
                batch_size=now_batch_size,
                shuffle=True,  # 每次训练打乱数据， 默认为False
                num_workers=0,  # 使用多进行程读取数据， 默认0，为不使用多进程, cuda需设为0
            )
            total_load_data_time += (time.time() - load_data_time)
            total_loss = 0
            step_num = len(loader)
            network_time = time.time()
            # 模型训练
            for step, (input_tensor, gt_tensor) in enumerate(loader):
                y_pred = model(input_tensor)
                loss = loss_fn(y_pred, gt_tensor)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss/step_num
            total_network_time += (time.time() - network_time)
            # tensorboard记录loss和lr数据, 打印各项信息
            tb.add_scalar('loss', avg_loss, idx)
            tb.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], idx)
            print("Epoch: ", epoch, "| Idx: ", idx,
                  "| Lr: {:.6f}".format(optimizer.state_dict()['param_groups'][0]['lr']),
                  "| Total_loss: {:.4f}".format(avg_loss), "| Get_data_time: {:.4f}".format(total_get_data_time),
                  "| Load_data_time: {:.4f}".format(total_load_data_time),
                  "| Network_time: {:.4f}".format(total_network_time))
            # # 保存loss最小的模型
            # if avg_loss < temp_loss:
            #     temp_loss = avg_loss
            #     print('--model is saved--')
            #     checkpoint = {  # 记录各种参数以方便断点续训
            #         "net": model.state_dict(),  # 网络参数
            #         'optimizer': optimizer.state_dict(),  # 优化器
            #         "epoch": epoch,  # 训练轮数
            #         'lr_schedule': scheduler.state_dict()  # lr如何变化
            #     }
            #     torch.save(checkpoint, model_file)
            # log_loss += avg_loss
        checkpoint = {  # 记录各种参数以方便断点续训
            "net": model.state_dict(),  # 网络参数
            'optimizer': optimizer.state_dict(),  # 优化器
            "epoch": epoch,  # 训练轮数
            'lr_schedule': scheduler.state_dict()  # lr如何变化
        }
        torch.save(checkpoint, model_file)
        if ((epoch + 1) % LR_ATTENUATION_EPOCHS) == 0:
            # lr衰减
            scheduler.step()
        if ((epoch + 1) % BATCH_SIZE_ATTENUATION_EPOCHS) == 0:
            # batch_size衰减
            now_batch_size >>= 1
        toatal_tb.add_scalar('loss', log_loss, epoch)
        toatal_tb.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

