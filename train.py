import time
import torch
from utils import *
import torch.utils.data
from random import shuffle
from prepare import Prepare
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, args, model):
        super(Trainer, self).__init__()
        self.args = args
        self.log_path = "./logs/%s/" % self.args.save_path
        self.model_path = "./checkpoints/%s/" % self.args.save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.train()
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.base_lr, weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.75)
        self.start_epoch = 0

    def load_pth(self):
        if self.args.resrme:
            checkpoint = torch.load(self.args.last_checkpoint)  # 加载断点
            self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            self.scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
            self.start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(self.start_epoch))
        else:
            print('无保存模型，将从头开始训练！')

    def pre_data(self):
        train_list = get_train_list(load_file_list(self.args.train_low_path), load_file_list(self.args.train_high_path))  # 载入数据集
        shuffle(train_list)
        PreData = Prepare(self.args)
        input_data, gt_data = PreData.prepare_nn_data(train_list)
        # 制作网络输入数据集
        input_tensor_data, gt_tensor_data = torch.from_numpy(input_data).to(self.device), \
                                            torch.from_numpy(gt_data).to(self.device)
        torch_dataset = torch.utils.data.TensorDataset(input_tensor_data, gt_tensor_data)  # 得到一个元组(x, y)
        loader = torch.utils.data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,  # 每次训练打乱数据， 默认为False
            num_workers=0,  # 使用多进行程读取数据， 默认0，为不使用多进程, cuda需设为0
        )
        return loader

    def train(self):
        self.load_pth()
        total_tb = SummaryWriter(self.log_path)
        for epoch in range(self.start_epoch, self.args.max_epoch):
            # 模型、日志保存路径
            model_file = os.path.join(self.model_path, "%s_%03d.pth" % (self.args.save_path, epoch))
            log_file = os.path.join(self.log_path, "%03d" % epoch)
            tb = SummaryWriter(log_file)
            temp_loss = 100000
            log_loss = 0
            for idx in range(10):
                total_get_data_time, total_network_time = 0, 0
                get_data_time = time.time()
                # 数据集处理
                loader = self.pre_data()
                total_get_data_time += (time.time() - get_data_time)
                total_loss = 0
                network_time = time.time()
                # 模型训练
                for step, (input_tensor, gt_tensor) in enumerate(loader):
                    y_pred = self.model(input_tensor)
                    loss = self.loss_fn(y_pred, gt_tensor)
                    total_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                avg_loss = total_loss / len(loader)
                total_network_time += (time.time() - network_time)
                # tensorboard记录loss和lr数据, 打印各项信息
                tb.add_scalar('loss', avg_loss, idx)
                tb.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], idx)
                print("Epoch: ", epoch, "| Idx: ", idx,
                      "| Lr: {:.6f}".format(self.optimizer.state_dict()['param_groups'][0]['lr']),
                      "| Total_loss: {:.4f}".format(avg_loss), "| Get_data_time: {:.4f}".format(total_get_data_time),
                      "| Network_time: {:.4f}".format(total_network_time))
                # 保存loss最小的模型
                if avg_loss < temp_loss and self.args.save_minloss_model:
                    temp_loss = avg_loss
                    print('--model is saved--')
                    checkpoint = {  # 记录各种参数以方便断点续训
                        "net": self.model.state_dict(),  # 网络参数
                        'optimizer': self.optimizer.state_dict(),  # 优化器
                        "epoch": epoch,  # 训练轮数
                        'lr_schedule': self.scheduler.state_dict()  # lr如何变化
                    }
                    torch.save(checkpoint, model_file)
                log_loss += avg_loss
            if not self.args.save_minloss_model:
                checkpoint = {  # 记录各种参数以方便断点续训
                    "net": self.model.state_dict(),  # 网络参数
                    'optimizer': self.optimizer.state_dict(),  # 优化器
                    "epoch": epoch,  # 训练轮数
                    'lr_schedule': self.scheduler.state_dict()  # lr如何变化
                }
                torch.save(checkpoint, model_file)
            if ((epoch + 1) % self.args.attenuation_epochs) == 0:
                # lr衰减
                self.scheduler.step()
            total_tb.add_scalar('loss', log_loss / 10, epoch)
            total_tb.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)
