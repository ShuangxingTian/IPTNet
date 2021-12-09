import random
import threading
from utils import *


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


class Prepare:
    def __init__(self, args):
        super(Prepare, self).__init__()
        self.args = args
        self.pre_batch_size = self.args.pre_batch_size
        self.gt_patch_size = self.args.gt_patch_size
        self.rec_patch_size = (self.args.gt_patch_size[0] // self.args.scale, self.args.gt_patch_size[1] // self.args.scale)

    def prepare_nn_data(self, train_list):
        thread_num = int(self.pre_batch_size / 8)
        batchSizeRandomList = random.sample(range(0, len(train_list)), thread_num)
        input_list = [0 for i in range(thread_num)]
        gt_list = [0 for i in range(thread_num)]
        t = []
        for i in range(thread_num):
            t.append(Reader(self.args, train_list[batchSizeRandomList[i]], i, input_list, gt_list, self.pre_batch_size,
                            self.gt_patch_size, self.rec_patch_size, thread_num))
        for i in range(thread_num):
            t[i].start()
        for i in range(thread_num):
            t[i].join()
        input_list = np.reshape(input_list, (self.pre_batch_size, 1, self.rec_patch_size[0], self.rec_patch_size[1]))
        gt_list = np.reshape(gt_list, (self.pre_batch_size, 1, self.gt_patch_size[0], self.gt_patch_size[1]))

        return input_list, gt_list


class Reader(threading.Thread):
    def __init__(self, args, file_name, id, input_list, gt_list, pre_batch_size, gt_patch_size, rec_patch_size, thread_num):
        super(Reader, self).__init__()
        self.id = id
        self.args = args
        self.gt_list = gt_list
        self.file_name = file_name
        self.input_list = input_list
        self.thread_num = thread_num
        self.gt_patch_size = gt_patch_size
        self.pre_batch_size = pre_batch_size
        self.rec_patch_size = rec_patch_size

    def run(self):
        input_image = c_getYdata(self.file_name[0])
        gt_image = c_getYdata(self.file_name[1])
        # qp=int(self.file_name[0].split("\\")[-2].split("qp")[1])
        in_ = []
        gt_ = []
        for j in range(self.pre_batch_size // self.thread_num):
            input_imgY, gt_imgY = cropIPT(self.args, input_image, gt_image, self.rec_patch_size[0], self.gt_patch_size[1], "ndarray")
            input_imgY = normalize(input_imgY)
            gt_imgY = normalize(gt_imgY)

            in_.append(input_imgY)
            gt_.append(gt_imgY)

        self.input_list[self.id] = in_
        self.gt_list[self.id] = gt_


def cropIPT(args, input_image, gt_image, patch_width, patch_height, img_type):
    global input_cropped, gt_cropped
    assert type(input_image) == type(gt_image), "types are different."
    if img_type == "ndarray":
        in_row_ind_rec = random.randint(0, input_image.shape[0] - patch_width)
        in_col_ind_rec = random.randint(0, input_image.shape[1] - patch_width)
        # in_row_ind_gt = random.randint(0, gt_image.shape[0] - patch_height)
        # in_col_ind_gt = random.randint(0, gt_image.shape[1] - patch_height)
        in_row_ind_gt = in_row_ind_rec * args.scale
        in_col_ind_gt = in_col_ind_rec * args.scale

        input_cropped = input_image[in_row_ind_rec:in_row_ind_rec + patch_width, in_col_ind_rec:in_col_ind_rec + patch_width]
        gt_cropped = gt_image[in_row_ind_gt:in_row_ind_gt + patch_height, in_col_ind_gt:in_col_ind_gt + patch_height]

    elif img_type == "Image":
        in_row_ind_rec = random.randint(0, input_image.size[0] - patch_width)
        in_col_ind_rec = random.randint(0, input_image.size[1] - patch_width)
        in_row_ind_gt = in_row_ind_rec * args.scale
        in_col_ind_gt = in_col_ind_rec * args.scale

        input_cropped = input_image.crop(
            box=(in_row_ind_rec, in_col_ind_rec, in_row_ind_rec + patch_width, in_col_ind_rec + patch_width))
        gt_cropped = gt_image.crop(
            box=(in_row_ind_gt, in_col_ind_gt, in_row_ind_gt + patch_height, in_col_ind_gt + patch_height))

    return input_cropped, gt_cropped