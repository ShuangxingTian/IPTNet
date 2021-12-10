import torch
from torch.utils.tensorboard import SummaryWriter
from prepare import prepare_test_data
from utils import *
from IPTNet import pth_to_pt


class Test:
    def __init__(self, args, model):
        super(Test, self).__init__()
        self.args = args
        self.testout_path = "./testout/%s/" % self.args.load_path
        self.model_path = "./checkpoints/%s/" % self.args.load_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval()
        self.temp_pth = ''
        self.item_max = [0.0, 0]
        self.pthfiles = [f for f in os.listdir(self.model_path)]
        self.test_start_epoch = self.args.test_start_epoch

    def test(self):
        or_y, gt_y, fileName_list = prepare_test_data([self.args.low_path, self.args.gt_path])        # 导入模型结构
        print(len(self.pthfiles))
        toatal_tb = SummaryWriter(self.testout_path)
        for pth in self.pthfiles[self.test_start_epoch:]:
            temp_epoch = pth.split('.pth')[0]
            epoch = int(temp_epoch.split('_')[-1])

            if epoch > 1000:
                break
            total_imgs = len(fileName_list)
            total_psnr = 0
            m_state_dict = torch.load(os.path.join(self.model_path, pth))
            self.model.load_state_dict(m_state_dict['net'])
            for i in range(total_imgs):
                imgY = or_y[i][0]
                gtY = gt_y[i] if gt_y else 0
                input_tensor = torch.Tensor(imgY).cuda()
                with torch.no_grad():
                    out = self.model(input_tensor)
                out = out.to('cpu').numpy() * 255
                if gt_y:
                    p = psnr255(out, gtY)
                    total_psnr += p
                    print(fileName_list[i], p)
            avg_psnr = total_psnr / total_imgs
            if avg_psnr > self.item_max[0]:
                self.item_max[0] = avg_psnr
                self.item_max[1] = epoch
                self.temp_pth = pth

            print("AVG_PSNR:%.3f\tepoch:%d\n" % (avg_psnr, epoch))
            toatal_tb.add_scalar('avg_psnr', avg_psnr, epoch)
        QP = os.path.basename(self.args.gt_path.split('_')[-1])
        print("QP:%s\tepoch: %d\tavg_max:%.4f\t" % (QP, self.item_max[1], self.item_max[0]))
        print(os.path.join(self.model_path, self.temp_pth))

        # pth_to_pt(os.path.join(MODEL_PATH, temp_Pth), os.path.join(OUT_MODEL_PATH, QP))
