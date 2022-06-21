import os
import logging
import re
import glob
import numpy as np
import math
import cv2
import torch
import torch.nn as nn
import psutil

from pathlib import Path


# Inherit from CCNet
def id2trainId(label, classes, reverse=False):
    ignore_label = 255
    id_to_trainid={-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    label_copy = label.copy()
    if reverse:
        for v, k in id_to_trainid.items():
            label_copy[label == k] = v
    else:
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v

    unique_values = np.unique(label_copy)
    unique_values = unique_values[unique_values != ignore_label] # ignore specific label
    max_val = max(unique_values)
    min_val = min(unique_values)
    if max_val > (classes - 1) or min_val < 0:
        LOGGER.info('Labels can take value between 0 and number of classes {}.'.format(classes-1))
        LOGGER.info('You have following values as class labels:')
        LOGGER.info(unique_values)
        if max_val > (classes - 1):
            LOGGER.info("max_val {} > classes {}".format(max_val, classes))
        if min_val < 0:
            LOGGER.info("min_val {} <".format(min_val))
        LOGGER.info('Exiting!!')
        exit()

    return label_copy


def get_palette(num_classes):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_classes
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def put_palette(label, num_classes, palette=None, path=None):
    from PIL import Image as PILImage
    if num_classes and palette is None:
        palette = get_palette(num_classes)
    output_im = PILImage.fromarray(label)
    output_im.putpalette(palette)
    output_im = output_im.convert('RGB') # convert P(8位像素，使用調色板映射到任何其他模式) to RGB

    if path:
        output_im.save(path)
    return np.array(output_im)


# Inherit from YOLOR
def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def increment_path(path, exist_ok=False, sep='-', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def check_suffix(file=None, suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def select_device(device='', batch_size=None, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            LOGGER.info(f"Find CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)")  # bytes to MB
    else:
        LOGGER.info('Find CPU')

    return torch.device('cuda:0' if cuda else 'cpu')


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def intersect_dicts(da, db):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and v.shape == db[k].shape}


# Inhert from ESPNetv2, where adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
# IOU result same as compute_ccnet_eval
class iouEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 0

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        if type(predict) is torch.Tensor:
            predict = predict.cpu().numpy()
        if type(gth) is torch.Tensor:
            gth = gth.cpu().numpy()
        predict = predict.flatten()
        gth = gth.flatten()

        epsilon = 0.00000001
        hist = self.compute_hist(predict, gth)
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIou = np.nanmean(per_class_iu)

        self.overall_acc +=overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.mIOU += mIou
        self.batchCount += 1

    def getMetric(self):
        overall_acc = self.overall_acc/self.batchCount
        per_class_acc = self.per_class_acc / self.batchCount
        per_class_iu = self.per_class_iu / self.batchCount
        mIOU = self.mIOU / self.batchCount

        return overall_acc, per_class_acc, per_class_iu, mIOU


def safety_cpu(max=20):
    cpu = psutil.virtual_memory().used / 1E9 # G
    if cpu > max:
        print(f'Warning : cpu usage {cpu} is bigger then {max}')
        input()


def xywh2xyxy(x): # xywh is center xy, width, height, xyxy is top-left point and down-right point
    dim = x.dim if isinstance(x, torch.Tensor) else x.ndim
    if dim == 1:
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[1] = x[1] - x[3] / 2  # top left x
        y[2] = x[2] - x[4] / 2  # top left y
        y[3] = x[1] + x[3] / 2  # bottom right x
        y[4] = x[2] + x[4] / 2  # bottom right y
    else:
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x): # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    dim = x.dim if isinstance(x, torch.Tensor) else x.ndim
    if dim == 1:
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = (x[0] + x[2]) / 2  # x center
        y[1] = (x[1] + x[3]) / 2  # y center
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
    else:
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def plot_xywh(img, labels, color=(128, 128, 128), txt_color=(255, 255, 255)):
    if len(labels) > 0:
        img = img.copy()
        if not img.flags["C_CONTIGUOUS"] or not img.flags["F_CONTIGUOUS"]:
            img = np.ascontiguousarray(img, dtype=np.uint8)
        img_h, img_w, ch = img.shape
        for (c, x, y, w, h) in labels:
            if x > 1 or y > 1 or w > 1 or h > 1:
                w_off = (w) / 2
                h_off = (h) / 2
                p1 = (int(x - w_off), int(y - h_off))
                p2 = (int(x + w_off), int(y + h_off))
            else:
                w_off = (img_w * w) / 2
                h_off = (img_h * h) / 2
                p1 = (int(img_w * x - w_off), int(img_h * y - h_off))
                p2 = (int(img_w * x + w_off), int(img_h * y + h_off))
            cv2.putText(img, str(int(c)), (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 1, txt_color)
            cv2.rectangle(img, p1, p2, color)
    return img


try:
    from utils.cityscapes import Create_Cityscapes
    from utils.kitti import Create_Kitti
    def create_dataloader(params, mode='train', rank=-1):
        if params.root.lower().find('cityscapes') > 0:
            dataset, dataloader = Create_Cityscapes(params, mode=mode, rank=-1)
        elif params.root.lower().find('kitti') > 0:
            dataset, dataloader = Create_Kitti(params, mode=mode, rank=-1)
        return dataset, dataloader
except:
    print('No create_dataloader')

if __name__ == '__main__':
    print(get_palette(255))