import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class CALayer(nn.Module):
    def __init__(self, channel, reduction=10):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv1 = nn.Conv2d(channel, channel // reduction, (1, 1), padding=0, bias=True)
        self.ca_conv2 = nn.Conv2d(channel // reduction, channel, (1, 1), padding=0, bias=True)

    def forward(self, x):
        y = self.avg_pool(x)
        y = F.relu(self.ca_conv1(y), inplace=True)
        y = torch.sigmoid(self.ca_conv2(y))
        return x * y


class RCAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_conv1 = nn.Conv2d(40, 256, (1, 1), (1, 1), padding='same', bias=True)
        self.res_conv2 = nn.Conv2d(256, 24, (1, 1), (1, 1), padding='same', bias=True)
        self.res_conv3 = nn.Conv2d(24, 40, (3, 3), (1, 1), padding='same', bias=True)
        self.CALayer = CALayer(40, 10)

    def forward(self, temp_tensor):
        x = F.relu(self.res_conv1(temp_tensor))
        x = self.res_conv2(x)
        x = self.res_conv3(x)
        x = self.CALayer(x)

        x += temp_tensor
        return x


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self):
        super(ResidualGroup, self).__init__()
        self.resblock0 = RCAB()
        self.resblock1 = RCAB()
        self.resblock2 = RCAB()
        self.resblock3 = RCAB()
        # self.resblock4 = RCAB()
        # self.resblock5 = RCAB()
        # self.resblock6 = RCAB()
        # self.resblock7 = RCAB()
        self.res_conv1 = nn.Conv2d(40, 40, (3, 3), (1, 1), padding='same', bias=True)

    def forward(self, x):
        res = self.resblock0(x)
        res = self.resblock1(res)
        res = self.resblock2(res)
        res = self.resblock3(res)
        # res = self.resblock4(res)
        # res = self.resblock5(res)
        # res = self.resblock6(res)
        # res = self.resblock7(res)
        res = self.res_conv1(res)
        res += x
        return res


class ResBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.res_conv1 = nn.Conv2d(40, 256, (1, 1), (1, 1), padding='same', bias=True)
        self.res_conv2 = nn.Conv2d(256, 24, (1, 1), (1, 1), padding='same', bias=True)
        self.res_conv3 = nn.Conv2d(24, 40, (3, 3), (1, 1), padding='same', bias=True)
        self.CALayer = CALayer(40, 10)

    def forward(self, temp_tensor):
        x = F.relu(self.res_conv1(temp_tensor))
        x = self.res_conv2(x)
        x = self.res_conv3(x)
        x = self.CALayer(x)
        x += temp_tensor
        return x


class IPTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.resblock0 = ResBlock()
        # self.resblock1 = ResBlock()
        # self.resblock2 = ResBlock()
        # self.resblock3 = ResBlock()
        # self.resblock4 = ResBlock()
        # self.resblock5 = ResBlock()
        # self.resblock6 = ResBlock()
        # self.resblock7 = ResBlock()
        self.rg1 = ResidualGroup()
        self.rg2 = ResidualGroup()
        self.rg3 = ResidualGroup()

        self.conv1 = nn.Conv2d(1, 40, (3, 3), (1, 1), padding='same', bias=True)

        self.conv2 = nn.Conv2d(40, 32, (1, 1), (1, 1), padding='same', bias=True)
        self.conv3 = nn.Conv2d(40, 32, (3, 3), (1, 1), padding='same', bias=True)

        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding='same', bias=True)
        self.Transpose1 = nn.ConvTranspose2d(64, 1, (2, 2), (2, 2), (0, 0), bias=False)

        self.conv5 = nn.Conv2d(1, 64, (3, 3), (1, 1), padding='same', bias=True)
        self.conv6 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding='same', bias=True)
        self.conv7 = nn.Conv2d(64, 1, (3, 3), (1, 1), padding='same', bias=True)

        self.Transpose2 = nn.ConvTranspose2d(1, 1, (2, 2), (2, 2), (0, 0), bias=False)
        self.conv8 = nn.Conv2d(1, 32, (3, 3), (1, 1), padding='same', bias=True)
        self.conv9 = nn.Conv2d(1, 32, (1, 1), (1, 1), padding='same', bias=True)

        self.conv10 = nn.Conv2d(64, 1, (3, 3), (1, 1), padding='same', bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_temp = x
        x = self.conv1(x)
        # x = self.resblock0(x)
        # x = self.resblock1(x)
        # x = self.resblock2(x)
        # x = self.resblock3(x)
        # x = self.resblock4(x)
        # x = self.resblock5(x)
        # x = self.resblock6(x)
        # x = self.resblock7(x)
        x = self.rg1(x)
        x = self.rg2(x)
        x = self.rg3(x)
        x1 = F.relu(self.conv2(x))
        x2 = F.relu(self.conv3(x))
        x = torch.cat((x1, x2), 1)
        x = F.relu(self.conv4(x))
        x = F.relu(self.Transpose1(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x_temp = F.relu(self.Transpose2(x_temp))
        x_temp1 = F.relu(self.conv8(x_temp))
        x_temp2 = F.relu(self.conv9(x_temp))
        x_temp = torch.cat((x_temp1, x_temp2), 1)
        x_temp = self.conv10(x_temp)
        x += x_temp

        return x


def test():
    net = IPTNet().cuda()
    summary(net, (1, 32, 32))


def pth_to_pt(path, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_model = IPTNet().to(device).eval()
    my_model.load_state_dict(torch.load(path)['net'], strict=False)
    with torch.no_grad():
        pt = torch.jit.trace(my_model, (torch.ones(1, 1, 2500, 1600).to(device)))
    pt.save(os.path.join(out_path, 'IPTNet.pt'))


if __name__ == '__main__':
    test()
    # out_path = "D:/tian/HM-16.9-pt/model/IPTNet-pt/qp22"
    # pth_to_pt('./checkpoints/HEVC_IPTNet_X2cnn_QP22_20211126/HEVC_IPTNet_X2cnn_QP22_20211124_171.pth', out_path)
