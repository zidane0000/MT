import argparse
import json
import os
import cv2
import random
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader


def get_sampler(dataset):
    if torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


class Kitti(Dataset):
    def __init__(
        self,
        file: str,        
        input_height: int,
        input_width: int,
        random_flip: bool = False,
        random_crop: bool = False,
        transform=None
    ) -> None:
        super(Kitti, self).__init__()
        self.file = file
        self.transform = transform
        self.height, self.width = input_height, input_width
        
        self.random_flip = random_flip
        self.random_crop = random_crop
        
        try:
            self.file = Path(self.file)
            with open(self.file) as f:
                f = f.read().splitlines()
                parent = str(self.file.parent) + os.sep
                self.images = [x.replace('./', parent) if x.startswith('./') else x for x in f]
            assert self.images, f'No images found in {self.file}'
        except Exception as e:
            raise Exception(f'Error loading data from {self.file}')

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        """
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        assert image is not None, f'No images found in {self.images[index]}'
                
        '''
        images : '../kitti/data_depth_annotated/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png'
        targets: '../kitti/data_depth_annotated/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png'
        '''
        depth_path = self.images[index].replace(self.images[index].split('/')[-5]+'/','')
        depth_path = depth_path.replace('/image_02/data','/proj_depth/groundtruth/image_02')
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)        
        assert depth is not None, f'No images found in {self.targets[index]}'
        
        if self.random_crop:
            image, depth = do_random_crop(image, depth, self.width, self.height)

        if self.random_flip:
            image, depth = do_random_flip(image, depth)

        if self.transform is not None:
            image = self.transform(image)
            depth = self.transform(depth)
        
        # (w, h, channel) -> (channel, w, h) and reszie if no random crop
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = np.moveaxis(image, -1, 0)

        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        return torch.from_numpy(image), torch.zeros(1), torch.from_numpy(depth)

    def __len__(self) -> int:
        return len(self.images)
        # return 50


def Create_Kitti(params, mode='train', rank=-1):
    batch_size = params.batch_size
    workers = params.workers
    input_height, input_width = params.input_height, params.input_width
    
    file = params.root + '/kitti_train.txt' if mode == 'train' else params.root + '/kitti_test.txt'
    dataset = Kitti(file,
                    input_height=input_height,
                    input_width=input_width,
                    random_flip=params.random_flip,
                    random_crop=params.random_crop)
    sampler = None if rank == -1 else get_sampler(dataset)
    dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=workers,
                    shuffle=True and sampler is None,
                    sampler=sampler,
                    pin_memory=True)
    
    return dataset, dataloader


def do_random_flip(image, target):
    up_down_flip = np.random.choice(2) * 2 - 1
    left_right_flip = np.random.choice(2) * 2 - 1
    image = image[::left_right_flip, ::up_down_flip].copy()
    target = target[::left_right_flip, ::up_down_flip].copy()
    return image, target


def do_random_crop(image, target, input_width, input_height):
    img_h, img_w, channel = image.shape
    
    h_gap, w_gap = img_h - input_height, img_w - input_width
    if h_gap < 0 or w_gap < 0:
        top = abs(int(h_gap / 2)) if h_gap < 0 else 0
        down = abs(h_gap + top) if h_gap < 0 else 0
        left = int(w_gap / 2) if w_gap < 0 else 0
        right = abs(w_gap + left) if w_gap < 0 else 0
        image = cv2.copyMakeBorder(image, top, down, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        target = cv2.copyMakeBorder(target, top, down, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        img_h, img_w, channel = image.shape
    
    h_off = random.randint(0, img_h - input_height)
    w_off = random.randint(0, img_w - input_width)
        
    image = image[h_off:h_off+input_height, w_off:w_off+input_width, :]
    target = target[h_off:h_off+input_height, w_off:w_off+input_width]
    return image, target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',           type=str, default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes', help='root for Cityscapes')
    parser.add_argument('--batch-size',     type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--workers',        type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--input_height',   type=int, help='input height', default=320)
    parser.add_argument('--input_width',    type=int, help='input width',  default=640)
    parser.add_argument('--random-flip',    action='store_true', help='flip the image and target')
    parser.add_argument('--random-crop',    action='store_true', help='crop the image and target')

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)
    params = parser.parse_args()
    train_dataset, train_loader = Create_Kitti(params, mode='train')

    pbar = tqdm(train_loader, total=len(train_loader))
    for i, item in enumerate(pbar):
        img, _, depth = item

        np_depth = depth[0].cpu().numpy()
        cv2.imwrite('depth.jpg', np_depth)

        heat_depth = (np_depth * 255).astype('uint8')
        heat_depth = cv2.applyColorMap(heat_depth, cv2.COLORMAP_JET)
        cv2.imwrite('heat.jpg', heat_depth)

        np_img = (img[0]).cpu().numpy().transpose(1,2,0)
        cv2.imwrite('img.jpg', np_img)
#         input()
