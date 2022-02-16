import argparse
import json
import os
import cv2
import numpy as np
from collections import namedtuple
import zipfile
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str

class Cityscapes(Dataset):
    def __init__(
            self,
            root: str,
            input_height: int,
            input_width: int,
            split: str = "train",
            mode: str = "fine",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(Cityscapes, self).__init__()
        self.root = root
        self.transforms = transforms
        self.height, self.width = input_height, input_width
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        ignore_label = 255
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

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
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "color", "disparity"))
         for value in self.target_type]

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
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))                
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Target is the image segmentation.
        """

        # image = Image.open(self.images[index]).convert('RGB')
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        image = np.moveaxis(image, -1, 0)

        targets: Any = []
        for i, t in enumerate(self.target_type):
            # target = Image.open(self.targets[index][i])
            target = cv2.imread(self.targets[index][i], cv2.IMREAD_GRAYSCALE)
            target = cv2.resize(target, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            if t == 'semantic':
                target =self.id2trainId(target)

            if t == 'disparity':
                target = np.expand_dims(target, axis=0) # (batch, h, w) => (batch, dims, h, w)
            
            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __len__(self) -> int:
        return len(self.images)

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
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'disparity':
            return 'disparity.png'


def Create_Cityscapes(params, mode='train'):
    batch_size = params.batch_size
    workers = params.workers
    input_height, input_width = params.input_height, params.input_width

    dataset = Cityscapes(params.root, input_height=input_height, input_width=input_width, split=mode, mode='fine', target_type=['semantic', 'disparity'])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    
    return dataset, dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',           type=str, default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes', help='root for Cityscapes')
    parser.add_argument('--batch-size',     type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--workers',        type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--input_height',   type=int, help='input height', default=480)
    parser.add_argument('--input_width',    type=int, help='input width',  default=640)
    params = parser.parse_args()
    train_dataset, train_loader = Create_Cityscapes(params, mode='train')

    for i, item in enumerate(train_loader):
        img, (smnt, depth) = item
        print(img.shape)
        print(smnt.shape)
        print(depth.shape)
        input()

