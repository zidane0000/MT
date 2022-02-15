import argparse
import numpy as np
import math
import os
import sys
import time
import torch

from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

from utils.cityscapes import Create_Cityscapes
from utils.general import one_cycle
from models.mt import MTmodel

def train(params):
    epochs = params.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = MTmodel(params).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=params.learning_rate)

    # Scheduler
    if params.linear_learning_rate:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - params.end_learning_rate) + params.end_learning_rate  # linear
    else:
        lf = one_cycle(1, params.end_learning_rate, epochs)  # cosine 1->params.end_learning_rate
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Dataset, DataLoader
    train_dataset, train_loader = Create_Cityscapes(params, mode='train')

    for epoch in range(epochs):
        model.train()

        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

        for i, item in pbar:
            img, (smnt, depth) = item
            img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            optimizer.zero_grad()
            output = model(img)
            loss = compute_loss(output, (smnt, depth))
            loss.backward()
            optimizer.step()
        
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes', help='root for Cityscapes')
    parser.add_argument('--epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--end_learning_rate', type=float, help='final OneCycleLR learning rate (lr0 * lrf)', default=1e-2)
    parser.add_argument('--linear-learning-rate', action='store_true', help='linear Learning Rate or cosine curve')
    params = parser.parse_args()

    train(params)