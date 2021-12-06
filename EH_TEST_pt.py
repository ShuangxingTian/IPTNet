import os
import numpy as np
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

from EH_UTILS_pt import *
from RWNet_pt import RWNet, pth_to_pt
# from RWNet_V1 import RWNet, pth_to_pt

EXP_DATA = "HEVC_RWNet_pt_QP37_20211129"

MODEL_PATH = "./checkpoints/%s/" % EXP_DATA
TESTOUT_PATH = "./testout/%s/" % EXP_DATA
OUT_DATA_PATH = "./outdata/%s/" % EXP_DATA

OUT_MODEL_PATH = r"D:\tian\HM-16.9-pt\model\RWNet-pt"
ORIGINAL_PATH = r"D:\tian\EHNet-pt\testSet\qp37"
GT_PATH = r"D:\tian\EHNet-pt\testSet\label"

TEST_START_EPOCH = 408

if __name__ == '__main__':
    temp_Pth = ''
    item_max = [0, 0]
    or_y, gt_y, fileName_list = prepare_test_data([ORIGINAL_PATH, GT_PATH])
    PthFiles = [f for f in os.listdir(MODEL_PATH)]
    # 导入模型结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RWNet().to(device)
    print(len(PthFiles))
    # 如果log文件夹不存在，则创建
    if not os.path.exists(TESTOUT_PATH):
        os.makedirs(TESTOUT_PATH)
    # 记录每一轮次的loss和lr
    toatal_tb = SummaryWriter(TESTOUT_PATH)
    for Pth in PthFiles[TEST_START_EPOCH:]:
        temp_epoch = Pth.split('.pth')[0]
        epoch = int(temp_epoch.split('_')[-1])

        if epoch > 1000:
            break
        total_imgs = len(fileName_list)
        total_psnr = 0
        # 导入模型
        m_state_dict = torch.load(os.path.join(MODEL_PATH, Pth))
        model.load_state_dict(m_state_dict['net'])
        # model.load_state_dict(m_state_dict)

        for i in range(total_imgs):
            imgY = or_y[i]
            gtY = gt_y[i] if gt_y else 0
            input_tensor = torch.Tensor(imgY).cuda()
            # 模型预测,输入测试集,输出预测结果
            with torch.no_grad():
                out = model(input_tensor)
            out = out.to('cpu').numpy() * 255
            if gt_y:
                p = psnr255(out, gtY)
                total_psnr += p
                print(fileName_list[i], p)
        avg_psnr = total_psnr / total_imgs
        if avg_psnr > item_max[0]:
            item_max[0] = avg_psnr
            item_max[1] = epoch
            temp_Pth = Pth

        print("AVG_PSNR:%.3f\tepoch:%d\n" % (avg_psnr, epoch))
        toatal_tb.add_scalar('avg_psnr', avg_psnr, epoch)
    QP = os.path.basename(ORIGINAL_PATH.split('_')[-1])
    print("QP:%s\tepoch: %d\tavg_max:%.4f\t" % (QP, item_max[1], item_max[0]))
    print(os.path.join(MODEL_PATH, temp_Pth))

    pth_to_pt(os.path.join(MODEL_PATH, temp_Pth), os.path.join(OUT_MODEL_PATH, QP))
