import argparse
import json
import os
import cv2
import random
import torch
import numpy as np

from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str

try:
    from .general import id2trainId, put_palette, plot_xywh, xywh2xyxy
except:
    from general import id2trainId, put_palette, plot_xywh, xywh2xyxy
    
    
def get_sampler(dataset):
    if torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None
    

class Cityscapes(Dataset):
    def __init__(
            self,
            root: str,
            input_height: int,
            input_width: int,
            num_classes: int,
            split: str = "train",
            mode: str = "fine",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            random_flip: bool = False,
            random_crop: bool = False,
    ) -> None:
        super(Cityscapes, self).__init__()
        self.root = root
        self.transform = transform
        self.height, self.width = input_height, input_width
        self.num_classes = num_classes
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.images = []
        self.targets = []

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]        

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                if file_name.find('.png') > -1: # Only accept png file
                    target_types = []
                    for t in self.target_type:
                        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], self._get_target_suffix(self.mode, t))
                        if t == 'label':
                            target_types.append(os.path.join(target_dir, target_name).replace('gtFine', 'labels'))
                        else:
                            target_types.append(os.path.join(target_dir, target_name))
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(target_types)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        """
        # image = Image.open(self.images[index]).convert('RGB')
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        
        smnt = cv2.imread(self.targets[index][0], cv2.IMREAD_GRAYSCALE)
        if self.targets[index][0].endswith('_labelIds.png'):
            smnt = id2trainId(smnt, self.num_classes)
        
        depth = cv2.imread(self.targets[index][1], cv2.IMREAD_GRAYSCALE)

        labels = np.zeros((0, 5), dtype=np.float32)
        label_path = self.targets[index][2]        
        if os.path.isfile(label_path):
            with open(label_path) as f:
                labels = [x.split() for x in f.read().strip().splitlines() if len(x)] # cls, x, y, w, h
                labels = np.array(labels, dtype=np.float32)
        
        if self.random_crop:
            image, (smnt, depth), labels = do_random_crop(image, [smnt, depth], labels, self.width, self.height)

        if self.random_flip:
            image, (smnt, depth), labels = do_random_flip(image, [smnt, depth], labels)

        if self.transform is not None:
            image = self.transform(image)
            (smnt, depth) = self.transform((smnt, depth))
            labels = self.transform(labels)
        
        # (w, h, channel) -> (channel, w, h) and reszie if no random crop
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = np.moveaxis(image, -1, 0)

        smnt = cv2.resize(smnt, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)                        
        
        return torch.from_numpy(image), torch.from_numpy(smnt), torch.from_numpy(depth), torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.images)
        # return 50

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelTrainIds.png'.format(mode) # labelTrainIds -> [0. - 18. 255.] -> id2trainId -> [0. 1. 2. 3. 4. 5. 255.]
            # return '{}_labelIds.png'.format(mode) # labelIds -> [0. - 33. -1.] -> id2trainID -> [0. - 18. 255.]
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'disparity':
            return 'disparity.png'
        elif target_type == 'label':
            return 'leftImg8bit.txt'


def collate_fn(batch):
    images, smnts, depths = zip(*batch)
    return torch.stack(images, 0), torch.stack(smnts, 0), torch.stack(depths, 0)


def Create_Cityscapes(params, mode='train', rank=-1):
    batch_size = params.batch_size
    workers = params.workers
    input_height, input_width = params.input_height, params.input_width

    dataset = Cityscapes(params.root,
                        input_height=input_height,
                        input_width=input_width,
                        num_classes=params.num_classes,
                        split=mode,
                        mode='fine',
                        target_type=['semantic', 'disparity', 'label'],
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


def do_random_flip(image, target, labels):
    up_down_flip = np.random.choice(2) * 2 - 1
    left_right_flip = np.random.choice(2) * 2 - 1
    image = image[::left_right_flip, ::up_down_flip].copy()

    for i in range(len(target)):
        target[i] = target[i][::left_right_flip, ::up_down_flip].copy()

    labels[:, 1] = labels[:, 1] if up_down_flip == 1 else 1 - labels[:, 1]
    labels[:, 2] = labels[:, 2] if left_right_flip == 1 else 1 - labels[:, 2]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',           type=str, default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes', help='root for Cityscapes')
    parser.add_argument('--batch-size',     type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--workers',        type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--input_height',   type=int, help='input height', default=640)
    parser.add_argument('--input_width',    type=int, help='input width',  default=1280)
    parser.add_argument('--random-flip',    action='store_true', help='flip the image and target')
    parser.add_argument('--random-crop',    action='store_true', help='crop the image and target')
    # Semantic Segmentation
    parser.add_argument('--num_classes',            type=int, help='Number of classes to predict (including background).', default=19)

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)
    params = parser.parse_args()
    train_dataset, train_loader = Create_Cityscapes(params, mode='train')

    pbar = tqdm(train_loader, total=len(train_loader))
    for i, item in enumerate(pbar):
        img, smnt, depth, labels = item
        
#         np_smnt = smnt[0].cpu().numpy()
#         np_smnt = id2trainId(np_smnt, 255, reverse=True)
#         np_smnt = put_palette(np_smnt, num_classes=255, path='smnt.jpg')
        
#         np_depth = depth[0].cpu().numpy()
#         cv2.imwrite('depth.jpg', np_depth)
            
#         heat_depth = (np_depth * 255).astype('uint8')
#         heat_depth = cv2.applyColorMap(heat_depth, cv2.COLORMAP_JET)
#         cv2.imwrite('heat.jpg', heat_depth)

        np_img = (img[0]).cpu().numpy().transpose(1,2,0)
        cv2.imwrite('img.jpg', np_img)

        np_img = plot_xywh(np_img, labels[0])
        cv2.imwrite('np_img.jpg', np_img)
        input()
