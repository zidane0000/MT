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
        transform: Optional[Callable] = None,
        augment:bool = False,
        random_hw: float = 0.5,
        random_flip: float = 0.5,
        random_crop: float = 0.5,
        multi_scale: float = 0.5, # Image will be scaled proportionally
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) -> None:
        super(Kitti, self).__init__()
        self.file = file
        self.transform = self.input_transform
        self.height, self.width = input_height, input_width

        self.augment = augment
        self.albumentations = Albumentations() if augment else None
        self.random_hw = random_hw
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.multi_scale = multi_scale
        self.mean = mean
        self.std = std

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
        
        smnt = np.zeros((0, 1), dtype=np.float32)
        
        '''
        images : '../kitti/data_depth_annotated/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png'
        targets: '../kitti/data_depth_annotated/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png'
        '''
        depth_path = self.images[index].replace(self.images[index].split('/')[-5]+'/','')
        depth_path = depth_path.replace('/image_02/data','/proj_depth/groundtruth/image_02')
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        assert depth is not None, f'No images found in {self.targets[index]}'

        labels = np.zeros((0, 5), dtype=np.float32)

        if self.augment:
            # Albumentations
            image = self.albumentations(image)

            # Multi-scale
            if random.random() < self.multi_scale:
                origin_h, origin_w = image.shape[:2]  # orig hw
                random_ratio = np.random.rand(1)
                new_w = int(self.width + (origin_w - self.width) * random_ratio)
                new_h = int(self.height + (origin_h - self.height) * random_ratio)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # smnt = cv2.resize(smnt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            if random.random() < self.random_crop:
                image, [depth], labels = do_random_crop(image, [depth], labels, self.width, self.height)

            if random.random() < self.random_flip:
                image, [depth], labels = do_random_flip(image, [depth], labels)

            # Brightness
            brightness = random.uniform(0.9, 1.1)
            image = image * brightness

        if self.transform is not None:
            image = self.transform(image)

        num_labels = len(labels)  # number of labels
        labels_out = torch.zeros((num_labels, 6))
        if num_labels:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # (w, h, channel) -> (channel, w, h) and reszie if no random crop
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = np.moveaxis(image, -1, 0)

        # smnt = cv2.resize(smnt, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        return torch.from_numpy(image), smnt, torch.from_numpy(depth), labels_out

    def __len__(self) -> int:
        return len(self.images)
        # return 50
    
    def input_transform(self, image, reverse=False):
        if reverse:
            image *= self.std
            image += self.mean
            image = image * 255.0
        else:
            image = image.astype(np.float32)
            image = image / 255.0
            image -= self.mean
            image /= self.std
        return image


def Create_Kitti(params, mode='train', rank=-1):
    batch_size = params.batch_size
    workers = params.workers
    input_height, input_width = params.input_height, params.input_width

    file = params.root + '/kitti_train.txt' if mode == 'train' else params.root + '/kitti_test.txt'
    dataset = Kitti(file,
                    input_height=input_height,
                    input_width=input_width,
                    augment=params.augment if mode=='train' else False,
                    random_hw=params.random_hw,
                    random_flip=params.random_flip,
                    random_crop=params.random_crop,
                    multi_scale=params.multi_scale)
    sampler = None if rank == -1 else get_sampler(dataset)
    dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=workers,
                    shuffle=True and sampler is None,
                    sampler=sampler,
                    pin_memory=True)

    return dataset, dataloader


def do_random_flip(image, target, labels):
    image = np.fliplr(image)
    # image = np.flipud(image)

    for i in range(len(target)):
        target[i] = np.fliplr(target[i])
        # target[i] = np.flipud(target[i])

    labels[:, 1] = 1 - labels[:, 1]
    # labels[:, 2] = 1 - labels[:, 2]
    return image, target, labels


def do_random_crop(image, target, labels, input_width, input_height):
    img_h, img_w, channel = image.shape
    h_off = random.randint(0, img_h - input_height)
    w_off = random.randint(0, img_w - input_width)
    image = image[h_off:h_off+input_height, w_off:w_off+input_width, :]

    for i in range(len(target)):
        target[i] = target[i][h_off:h_off+input_height, w_off:w_off+input_width]

    need_delete = []
    for i in range(len(labels)):
        label = labels[i]
        xyxy_label = xywh2xyxy(np.copy(label))
        left = xyxy_label[1] * img_w
        top = xyxy_label[2] * img_h
        right = xyxy_label[3] * img_w
        down = xyxy_label[4] * img_h

        if left < w_off or top > h_off+input_height or right > w_off+input_width or down < h_off:
            need_delete.append(i)
        else:
            # 等比例縮放
            label[1] = (label[1] * img_w - w_off) / input_width
            label[2] = (label[2] * img_h - h_off) / input_height
            label[3] = label[3] * img_w / input_width
            label[4] = label[4] * img_h / input_height
    labels = np.delete(labels, need_delete, axis=0)

    return image, target, labels


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A

            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.CLAHE(p=0.01),]  # transforms
            self.transform = A.Compose(T)

            print('albumentations: ' + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            print('no import albumentations')
            pass
        except Exception as e:
            print(f'albumentations: + {e}')

    def __call__(self, im, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im = new['image']
        return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',           type=str, default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes', help='root for Cityscapes')
    parser.add_argument('--batch-size',     type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--workers',        type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--input_height',   type=int, help='input height', default=320)
    parser.add_argument('--input_width',    type=int, help='input width',  default=640)
    
    # Augment
    parser.add_argument('--augment',        action='store_true', help='set for open augment')
    parser.add_argument('--random-hw',      type=float, default=0.0, help='random h and w in training')
    parser.add_argument('--random-flip',    type=float, default=0.5, help='flip the image and target')
    parser.add_argument('--random-crop',    type=float, default=0.5, help='crop the image and target')
    parser.add_argument('--multi-scale',    type=float, default=0.5, help='Image will be scaled proportionally')

    # Semantic Segmentation
    parser.add_argument('--smnt_num_classes',            type=int, help='Number of classes to predict (including background).', default=19)

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)

    # Object detection
    parser.add_argument('--obj_num_classes',        type=int, help='Number of classes to predict (including background) for object detection.', default=80)
    params = parser.parse_args()
    train_dataset, train_loader = Create_Kitti(params, mode='train')

    pbar = tqdm(train_loader, total=len(train_loader))
    for i, item in enumerate(pbar):
        img, smnt, depth, labels = item
        
#         print(img.shape)
#         print(depth.shape)
#         np_depth = depth[0].cpu().numpy()
#         cv2.imwrite('depth.jpg', np_depth)

#         heat_depth = (np_depth * 255).astype('uint8')
#         heat_depth = cv2.applyColorMap(heat_depth, cv2.COLORMAP_JET)
#         cv2.imwrite('heat.jpg', heat_depth)

#         np_img = (img[0]).cpu().numpy().transpose(1,2,0)
#         cv2.imwrite('img.jpg', np_img)
#         input()
