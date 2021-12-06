import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_conv1 = nn.Conv2d(40, 256, (1, 1), (1, 1), padding='same', bias=True)
        self.res_conv2 = nn.Conv2d(256, 24, (1, 1), (1, 1), padding='same', bias=True)
        self.res_conv3 = nn.Conv2d(24, 40, (3, 3), (1, 1), padding='same', bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)

    def forward(self, temp_tensor):
        x = F.relu(self.res_conv1(temp_tensor))
        x = self.res_conv2(x)
        x = self.res_conv3(x)

        x += temp_tensor
        return x


class RWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblock0 = ResBlock()
        self.resblock1 = ResBlock()
        self.resblock2 = ResBlock()
        self.resblock3 = ResBlock()
        self.resblock4 = ResBlock()
        self.resblock5 = ResBlock()
        self.resblock6 = ResBlock()
        self.resblock7 = ResBlock()

        self.conv1 = nn.Conv2d(1, 40, (3, 3), (1, 1), padding='same', bias=True)
        self.conv2 = nn.Conv2d(40, 1, (3, 3), (1, 1), padding='same', bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock0(out)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)
        out = self.resblock6(out)
        out = self.resblock7(out)
        out = self.conv2(out)
        out += x
        return out


def test():
    net = RWNet().cuda()
    summary(net, (1, 64, 64))


def pth_to_pt(path, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_model = RWNet().to(device)
    my_model.load_state_dict(torch.load(path)['net'], strict=False)
    with torch.no_grad():
        pt = torch.jit.trace(my_model, (torch.ones(1, 1, 2500, 1600).to(device)))
    pt.save(os.path.join(out_path, 'RWNet.pt'))


if __name__ == '__main__':
    test()
    # out_path = "D:/tian/HM-16.9-pt/model/IPTNet-pt/qp22"
    # pth_to_pt('./checkpoints/HEVC_IPTNet_X2cnn_QP22_20211126/HEVC_IPTNet_X2cnn_QP22_20211124_171.pth', out_path)
