import argparse

from utils import Cityscapes
from models import MTmodel

def train(params):
    # Model
    model = MTmodel(params)

    # Optimizer

    # Dataloader
    # https://pytorch.org/vision/0.8/datasets.html#cityscapes    
    train_dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                     target_type=['semantic', 'polygon', 'color']) # target_type = instance, semantic, polygon or color

    train_dataloader = Dataloader(train_dataset, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    params = parser.parse_args()

    train(params)