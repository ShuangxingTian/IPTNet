import math
import os
import random
import threading
import numpy as np
from PIL import Image
from IPT_TRAIN_pt import GT_PATCH_SIZE
from IPT_TRAIN_pt import PRE_BATCH_SIZE
from IPT_TRAIN_pt import REC_PATCH_SIZE


def prepare_nn_data(train_list):
    thread_num = int(PRE_BATCH_SIZE / 8)
    # 从train_list中随机抽取thread_num个图片
    batchSizeRandomList = random.sample(range(0, len(train_list)), thread_num)
    input_list = [0 for i in range(thread_num)]
    gt_list = [0 for i in range(thread_num)]
    t = []
    for i in range(thread_num):
        t.append(Reader(train_list[batchSizeRandomList[i]], i, input_list, gt_list, PRE_BATCH_SIZE, thread_num))
    for i in range(thread_num):
        t[i].start()
    for i in range(thread_num):
        t[i].join()
    input_list = np.reshape(input_list, (PRE_BATCH_SIZE, 1, REC_PATCH_SIZE[0], REC_PATCH_SIZE[1]))
    gt_list = np.reshape(gt_list, (PRE_BATCH_SIZE, 1, GT_PATCH_SIZE[0], GT_PATCH_SIZE[1]))

    return input_list, gt_list


def prepare_test_data(fileOrDir):
    original_ycbcr = []
    gt_y = []
    fileName_list = []
    if len(fileOrDir) == 2:

        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_train_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:
            or_imgY = c_getYdata(pair[0])
            gt_imgY = c_getYdata(pair[1])

            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, 1, or_imgY.shape[0], or_imgY.shape[1]))
            gt_imgY = np.resize(gt_imgY, (1, 1, gt_imgY.shape[0], gt_imgY.shape[1]))

            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list


class Reader(threading.Thread):
    def __init__(self, file_name, id, input_list, gt_list, BATCH_SIZE, thread_num):
        super(Reader, self).__init__()
        self.file_name = file_name
        self.id = id
        self.input_list = input_list
        self.gt_list = gt_list
        self.PRE_BATCH_SIZE = BATCH_SIZE
        self.thread_num = thread_num

    def run(self):
        input_image = c_getYdata(self.file_name[0])
        gt_image = c_getYdata(self.file_name[1])
        # qp=int(self.file_name[0].split("\\")[-2].split("qp")[1])
        in_ = []
        gt_ = []
        for j in range(self.PRE_BATCH_SIZE // self.thread_num):
            input_imgY, gt_imgY = cropIPT(input_image, gt_image, REC_PATCH_SIZE[0], GT_PATCH_SIZE[1], "ndarray")
            input_imgY = normalize(input_imgY)
            gt_imgY = normalize(gt_imgY)

            in_.append(input_imgY)
            gt_.append(gt_imgY)

        self.input_list[self.id] = in_
        self.gt_list[self.id] = gt_


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


def cropIPT(input_image, gt_image, patch_width, patch_height, img_type):
    global input_cropped, gt_cropped
    assert type(input_image) == type(gt_image), "types are different."
    if img_type == "ndarray":
        in_row_ind_rec = random.randint(0, input_image.shape[0] - patch_width)
        in_col_ind_rec = random.randint(0, input_image.shape[1] - patch_width)
        # in_row_ind_gt = random.randint(0, gt_image.shape[0] - patch_height)
        # in_col_ind_gt = random.randint(0, gt_image.shape[1] - patch_height)
        in_row_ind_gt = in_row_ind_rec * 2
        in_col_ind_gt = in_col_ind_rec * 2

        input_cropped = input_image[in_row_ind_rec:in_row_ind_rec + patch_width,
                        in_col_ind_rec:in_col_ind_rec + patch_width]
        gt_cropped = gt_image[in_row_ind_gt:in_row_ind_gt + patch_height, in_col_ind_gt:in_col_ind_gt + patch_height]

    elif img_type == "Image":
        in_row_ind_rec = random.randint(0, input_image.size[0] - patch_width)
        in_col_ind_rec = random.randint(0, input_image.size[1] - patch_width)
        in_row_ind_gt = in_row_ind_rec * 2
        in_col_ind_gt = in_col_ind_rec * 2

        input_cropped = input_image.crop(
            box=(in_row_ind_rec, in_col_ind_rec, in_row_ind_rec + patch_width, in_col_ind_rec + patch_width))
        gt_cropped = gt_image.crop(
            box=(in_row_ind_gt, in_col_ind_gt, in_row_ind_gt + patch_height, in_col_ind_gt + patch_height))

    return input_cropped, gt_cropped


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
