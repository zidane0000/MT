import argparse
import numpy as np
import math
import os
import sys
import time
import torch
import cv2

from pathlib import Path
from datetime import datetime
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

from utils.cityscapes import Create_Cityscapes
from utils.general import one_cycle, increment_path, select_device
from utils.loss import ComputeLoss
from models.mt import MTmodel

def train(params):
    epochs = params.epochs
    device = select_device(params.device)

    # Directories
    save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok, mkdir=True)
    print("saving to " + str(save_dir))

    # Model
    print("begin to bulid up model...")
    model = MTmodel(params)
    if device != 'cpu' and torch.cuda.device_count() > 1:
        print("use multi-gpu, device=" + params.device)
        device_ids = [int(i) for i in params.device.split(',')]
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    model = model.to(device)
    print("load model to device")

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
    test_dataset, test_loader = Create_Cityscapes(params, mode='val') # semantic no test data

    # loss
    compute_loss = ComputeLoss()

    for epoch in range(epochs):
        # Train
        model.train()
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for i, item in pbar:
            img, (smnt, depth) = item
            img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            smnt = smnt.to(device)
            depth = depth.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss, (smnt_loss, depth_loss) = compute_loss(output, (smnt, depth))
            loss.backward()
            optimizer.step()

            # log
            mem = f'{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('epoch : %4s  ' + 'mem : %4s  ' + '     semantic : %4.4g  ' + '     depth : %4.4g') % (
                    f'{epoch}/{epochs - 1}', mem, smnt_loss, depth_loss))        
        scheduler.step()

        # Test
        model.eval()
        test_bar = enumerate(test_loader)
        test_bar = tqdm(test_bar, total=len(test_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for i, item in test_bar:
            img, (smnt, depth) = item
            img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            smnt = smnt.to(device)
            depth = depth.to(device)

            with torch.no_grad():
                output = model(img)
                loss, (smnt_loss, depth_loss) = compute_loss(output, (smnt, depth))            

            # log
            mem = f'{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            test_bar.set_description(('epoch : %4s  ' + 'mem : %4s  ' + 'test-semantic : %4.4g  ' + 'test-depth : %4.4g') % (
                    f'{epoch}/{epochs - 1}', mem, smnt_loss, depth_loss))

        # Save model
        if (epoch % params.save_cycle) == 0:
            ckpt = {'epoch': epoch,
                    'model': model.state_dict(),
                    'date': datetime.now().isoformat()}
            torch.save(ckpt, save_dir / 'epoch-{}.pt'.format(epoch))
            del ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',               type=str, help='root for Cityscapes', default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes')
    parser.add_argument('--project',            type=str, help='directory to save checkpoints and summaries', default='./runs/train/')
    parser.add_argument('--name',               type=str, help='save to project/name', default='mt')
    parser.add_argument('--epochs',             type=int, help='number of epochs', default=50)
    parser.add_argument('--save-cycle',         type=int, help='save when cycle', default=10)
    parser.add_argument('--batch-size',         type=int, help='total batch size for all GPUs', default=8)
    parser.add_argument('--workers',            type=int, help='maximum number of dataloader workers', default=8)
    parser.add_argument('--input_height',       type=int,   help='input height', default=256)
    parser.add_argument('--input_width',        type=int,   help='input width',  default=512)
    parser.add_argument('--learning_rate',      type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--end_learning_rate',  type=float, help='final OneCycleLR learning rate (lr0 * lrf)', default=1e-2)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--linear-learning-rate', action='store_true', help='linear Learning Rate or cosine curve')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plot', action='store_true', help='plot the loss and eval result')
    parser.add_argument('--random-flip', action='store_true', help='flip the image and target')
    parser.add_argument('--random-crop', action='store_true', help='crop the image and target')
    params = parser.parse_args()

    train(params)