import torch
from test import Test
from utils import *
from IPTNet import IPTNet
from option import args
from train import Trainer

if __name__ == '__main__':
    Create_folder(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IPTNet(args).to(device)
    if args.istrain:
        train = Trainer(args, model)
        train.train()
    if args.istest:
        test = Test(args, model)
        test.test()
