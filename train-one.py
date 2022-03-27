# -*- coding: utf-8 -*-
import argparse
import numpy as np
import math
import os
import sys
import time
import torch
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

from val import val_one as val
from utils.cityscapes import Create_Cityscapes
from utils.general import one_cycle, increment_path, select_device, LOGGER
from utils.loss import CriterionDSN as ComputeLoss
from models.decoder.espnet import ESPNet as OneModel

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(params):
    epochs = params.epochs
    device = select_device(params.device, batch_size=params.batch_size)
    save_dir = params.save_dir
    cuda = device.type != 'cpu'
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert params.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        torch.distributed.init_process_group(backend="nccl" if torch.distributed.is_nccl_available() else "gloo", init_method='env://')

    # Model
    LOGGER.info("begin to bulid up model...")
    model = OneModel(classes=19).to(device)
    LOGGER.info("load model to device")

    # Optimizer
    if params.adam:
        optimizer = Adam(model.parameters(), lr=params.learning_rate, betas=(params.momentum, 0.999), weight_decay=params.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
    LOGGER.info(f"optimizer : {type(optimizer).__name__}")

    # Scheduler
    if params.linear_learning_rate:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - params.end_learning_rate) + params.end_learning_rate  # linear
    else:
        lf = one_cycle(1, params.end_learning_rate, epochs)  # cosine 1->params.end_learning_rate
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.info('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.')
        model = torch.nn.DataParallel(model)
        
    # SyncBatchNorm
    if params.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
        
    # DDP mode
    '''
    多卡訓練的模型設置： 
        最主要的是find_unused_parameters和broadcast_buffers參數； 
        find_unused_parameters：如果模型的輸出有不需要進行反傳的(比如部分參數被凍結/或者網絡前傳是動態的)，設置此參數為True;如果你的代碼運行後卡住某個地方不動，基本上就是該參數的問題。
        broadcast_buffers：設置為True時，在模型執行forward之前，gpu0會把buffer中的參數值全部覆蓋到別的gpu上。注意這和同步BN並不一樣，同步BN應該使用SyncBatchNorm。
    '''
    if cuda and RANK != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        LOGGER.info('Using DDP')
        
    # Dataset, DataLoader
    train_dataset, train_loader = Create_Cityscapes(params, mode='train', rank=LOCAL_RANK)

    # loss
    compute_loss = ComputeLoss()
    loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
            
        # Train
        model.train()
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]: # Process 0
            pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar        

        # mean loss
        mean_loss = torch.zeros(1, device=device)
        for i, item in pbar:
            img, (smnt, depth) = item
            img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            smnt = smnt.to(device)
            # depth = depth.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss = compute_loss(output, smnt)
            
            if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            loss.backward()
            optimizer.step()

            # Log
            if RANK in [-1, 0]: # Process 0
                mem = f'{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                mean_loss = (mean_loss * i + loss) / (i + 1)
                pbar.set_description(('epoch:%8s' + '  mem:%8s' + '  loss:%6.6g') % (f'{epoch}/{epochs - 1}', mem, mean_loss))
        scheduler.step()
        
        if RANK in [-1, 0]: # Process 0
            # Test
            val_loss, _, _ = val(params, save_dir=save_dir, model=model, device=device, compute_loss=compute_loss, task = "smnt")
            
            loss_history.append(mean_loss)
            val_loss_history.append(val_loss)

            # Save model
            if (epoch % params.save_cycle) == 0:
                ckpt = {'epoch': epoch,
                        'model': model.state_dict(),
                        'date': datetime.now().isoformat()}
                torch.save(ckpt, save_dir / 'epoch-{}.pt'.format(epoch))
                del ckpt
    
    if RANK in [-1, 0]: # Process 0
        plt.plot(range(epochs), loss_history)
        plt.plot(range(epochs), loss_history)
        plt.legend(['train','val'])
        plt.savefig(save_dir / 'history.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',               type=str, help='root for Cityscapes', default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes')
    parser.add_argument('--project',            type=str, help='directory to save checkpoints and summaries', default='./runs/train/')
    parser.add_argument('--name',               type=str, help='save to project/name', default='mt')
    parser.add_argument('--encoder',            type=str, help='Choose Encoder in MT', default='densenet161')
    parser.add_argument('--epochs',             type=int, help='number of epochs', default=50)
    parser.add_argument('--save-cycle',         type=int, help='save when cycle', default=10)
    parser.add_argument('--batch-size',         type=int, help='total batch size for all GPUs', default=8)
    parser.add_argument('--workers',            type=int, help='maximum number of dataloader workers', default=8)
    parser.add_argument('--input_height',       type=int,   help='input height', default=256)
    parser.add_argument('--input_width',        type=int,   help='input width',  default=512)
    parser.add_argument('--local_rank',         type=int,   help='DDP parameter, do not modify', default=-1)
    parser.add_argument("--momentum",           type=float, help="Momentum component of the optimiser.", default=0.937)
    parser.add_argument('--weight_decay',       type=float, help='weight decay factor for optimization', default=1e-2)
    parser.add_argument('--learning_rate',      type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--end_learning_rate',  type=float, help='final OneCycleLR learning rate (lr0 * lrf)', default=1e-2)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--linear-learning-rate', action='store_true', help='linear Learning Rate or cosine curve')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plot', action='store_true', help='plot the loss and eval result')
    parser.add_argument('--random-flip', action='store_true', help='flip the image and target')
    parser.add_argument('--random-crop', action='store_true', help='crop the image and target')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')

    # Semantic Segmentation
    parser.add_argument('--num_classes',            type=int, help='Number of classes to predict (including background).', default=19)

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)

    params = parser.parse_args()    
    
    # Directories
    params.save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok) # if create here will cause 2 folder
    params.save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("saving to " + str(params.save_dir))
    
    train(params)
