import os
import math
import numpy as np
from PIL import Image


def c_getYdata(path):
    return getYdata(path, getWH(path))


def getWH(yuvfileName):  # Test
    w_included, h_included = os.path.splitext(os.path.basename(yuvfileName))[0].split('x')
    w = w_included.split('_')[-1]
    h = h_included.split('_')[0]
    return int(w), int(h)


def getYdata(path, size):
    w = size[0]
    h = size[1]
    Yt = np.zeros([h, w], dtype="uint8", order='C')
    with open(path, 'rb') as fp:
        fp.seek(0, 0)
        Yt = fp.read()
        tem = Image.frombytes('L', [w, h], Yt)
        Yt = np.asarray(tem, dtype='float32')
    return Yt


def normalize(x):
    x = x / 255.
    return truncate(x, 0., 1.)


def denormalize(x):
    x = x * 255.
    return truncate(x, 0., 255.)


def truncate(input, min, max):
    input = np.where(input > min, input, min)
    input = np.where(input < max, input, max)
    return input


def load_file_list(directory):
    list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.split(".")[-1] == "yuv":
                list.append(file_name)
    return sorted(list)


def get_train_list(lowList, highList):
    assert len(lowList) == len(highList), "low:%d, high:%d" % (len(lowList), len(highList))
    train_list = []
    for i in range(len(lowList)):
        train_list.append([lowList[i], highList[i]])
    return train_list


def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr255(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def Create_folder(args):
    LOG_PATH = "./logs/%s/" % args.save_path  # log路径
    MODEL_PATH = "./checkpoints/%s/" % args.save_path  # 存储训练好的模型
    TESTOUT_PATH = "./testout/%s/" % args.load_path
    if not os.path.exists(LOG_PATH):  # 如果log和model文件夹不存在，则创建文件夹
        os.makedirs(LOG_PATH)
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
    if not os.path.exists(TESTOUT_PATH):
        os.makedirs(TESTOUT_PATH)
