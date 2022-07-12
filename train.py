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

from val import val
from utils.cityscapes import Create_Cityscapes
from utils.general import one_cycle, increment_path, select_device, LOGGER, safety_cpu
from utils.loss import ComputeLoss
from models.mt import MTmodel

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
    model = MTmodel(params).to(device)
    LOGGER.info("load model to device")

    # Pretrained
    if params.weights != None:
        print('=> loading pretrained model {}'.format(params.weights))
        ckpt = torch.load(params.weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model_dict = model.state_dict()
        pretrained_dict = ckpt['model']
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()} # remove if save by parallel
        LOGGER.info(f'inhert : {len(set(model_dict) & set(pretrained_dict))} / {len(set(pretrained_dict))}')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Optimizer
    if params.adam:
        optimizer = Adam([{'params': model.parameters()}],
                         lr=params.learning_rate, betas=(params.momentum, 0.999), eps=1e-08, weight_decay=params.weight_decay)
    else:
        optimizer = SGD([{'params': model.parameters(), 'weight_decay': params.weight_decay}],
                         lr=params.learning_rate, momentum=params.momentum)
    LOGGER.info(f"optimizer : {type(optimizer).__name__}")

    # Dataset, DataLoader
    train_dataset, train_loader = Create_Cityscapes(params, mode='train', rank=LOCAL_RANK)

    # loss
    compute_loss = ComputeLoss(model)
    smnt_loss_history = []
    depth_loss_history = []
    obj_loss_history = []
    smnt_val_loss_history = []
    depth_val_loss_history = []
    obj_val_loss_history = []

    # Val
    mean_iou_history = []
    d1_history = []
    d2_history = []
    d3_history = []
    map50_history = []
    map5095_history = []

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

    # Warm-Up
    num_batches = len(train_loader)
    num_warmup_iter = params.warmup * num_batches
    warmup_bias_lr = 0.1  # warmup initial bias lr
    warmup_momentum = 0.8  # warmup initial momentum

    for epoch in range(epochs):
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        # Train
        model.train()
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]: # Process 0
            pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

        # Strategy
        if params.strategy:
            from strategy import Strategy3 as Strategy
            model, compute_loss = Strategy(params, epoch, epochs, model)

        # mean loss
        mean_smnt_loss = torch.zeros(1, device=device)
        mean_depth_loss = torch.zeros(1, device=device)
        mean_obj_loss = torch.zeros(1, device=device)
        for i, item in pbar:
            img, smnt, depth, labels = item
            img = img.to(device, non_blocking=True)
            smnt = smnt.to(device)
            depth = depth.to(device)
            labels = labels.to(device)

            # Warm-up
            ni = (i + num_batches * epoch)
            if ni <= num_warmup_iter:
                xi = [0, num_warmup_iter]  # x interp
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [warmup_momentum, params.momentum])

            optimizer.zero_grad()

            output = model(img)

            loss, spilt_loss = compute_loss(output, (smnt, depth, labels))

            if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            loss.backward()
            optimizer.step()

            # Log
            if RANK in [-1, 0]: # Process 0
                safety_cpu(params.max_cpu)
                mem = f'{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)

                log = ('epoch:%8s' + '  mem:%8s') % (f'{epoch}/{epochs - 1}', mem)
                task = 0
                if params.semantic_head != '':
                    smnt_loss = spilt_loss[task]
                    task+=1
                    mean_smnt_loss = (mean_smnt_loss * i + smnt_loss) / (i + 1)
                    log += ('      semantic:%6.6g') % (mean_smnt_loss)

                if params.depth_head != '':
                    depth_loss = spilt_loss[task]
                    task+=1
                    mean_depth_loss = (mean_depth_loss * i + depth_loss) / (i + 1)
                    log += ('      depth:%6.6g') % (mean_depth_loss)

                if params.obj_head != '':
                    obj_loss = spilt_loss[task]
                    task+=1
                    mean_obj_loss = (mean_obj_loss * i + obj_loss) / (i + 1)
                    log += ('      obj:%6.6g') % (mean_obj_loss)

                pbar.set_description(log)
        scheduler.step()

        if RANK in [-1, 0]: # Process 0
            # Test
            all_val_loss, all_val = val(params, save_dir=save_dir, model=model, device=device, compute_loss=compute_loss)

            task = 0
            if params.semantic_head != '':
                smnt_val_loss = all_val_loss[task]
                (smnt_mean_iou_val, smnt_iou_array_val) = all_val[task]
                task+=1
                smnt_loss_history.append(mean_smnt_loss.detach().cpu().numpy())
                smnt_val_loss_history.append(smnt_val_loss.detach().cpu().numpy())
                mean_iou_history.append(smnt_mean_iou_val)


            if params.depth_head != '':
                depth_val_loss = all_val_loss[task]
                depth_val = all_val[task]
                task+=1
                depth_loss_history.append(mean_depth_loss.detach().cpu().numpy())
                depth_val_loss_history.append(depth_val_loss.detach().cpu().numpy())
                d1_history.append(depth_val[-3])
                d2_history.append(depth_val[-2])
                d3_history.append(depth_val[-1])

            if params.obj_head != '':
                obj_val_loss = all_val_loss[task]
                obj_map50, obj_map5095 = all_val[task]
                task+=1
                obj_loss_history.append(mean_obj_loss.detach().cpu().numpy())
                obj_val_loss_history.append(obj_val_loss.detach().cpu().numpy())
                map50_history.append(obj_map50)
                map5095_history.append(obj_map5095)

            if params.strategy:
                max_len = max([len(smnt_loss_history), len(depth_loss_history), len(obj_loss_history)])
                if params.semantic_head == '':
                    smnt_loss_history.append(None)
                    smnt_val_loss_history.append(None)
                    mean_iou_history.append(None)
                if params.depth_head == '':
                    depth_loss_history.append(None)
                    depth_val_loss_history.append(None)
                    d1_history.append(None)
                    d2_history.append(None)
                    d3_history.append(None)
                if params.obj_head == '':
                    obj_loss_history.append(None)
                    obj_val_loss_history.append(None)
                    map50_history.append(None)
                    map5095_history.append(None)

            # Save model
            if (epoch % params.save_cycle) == 0:
                ckpt = {'epoch': epoch,
                        'model': model.state_dict(),
                        'date': datetime.now().isoformat()}
                torch.save(ckpt, save_dir / 'epoch-{}.pt'.format(epoch))
                del ckpt

    if RANK in [-1, 0]: # Process 0
        loss_history_fig = plt.figure(0)
        tmp_fig = plt.figure(1)
        loss_history = loss_history_fig.add_subplot(1, 1, 1)
        legend = []
        if params.semantic_head != '':
            loss_history.plot(range(epochs), smnt_loss_history)
            loss_history.plot(range(epochs), smnt_val_loss_history)
            legend += ['semantic','semantic(val)']

            sub_fig = tmp_fig.add_subplot(1, 1, 1)
            sub_fig.plot(range(epochs), mean_iou_history)
            tmp_fig.savefig(save_dir / 'mean_iou_history.png')
            tmp_fig.clf()

        if params.depth_head != '':
            loss_history.plot(range(epochs), depth_loss_history)
            loss_history.plot(range(epochs), depth_val_loss_history)
            legend += ['depth','depth(val)']

            sub_fig = tmp_fig.add_subplot(1, 1, 1)
            sub_fig.plot(range(epochs), d1_history)
            sub_fig.plot(range(epochs), d2_history)
            sub_fig.plot(range(epochs), d3_history)
            sub_fig.legend(['d1','d2','d3'])
            tmp_fig.savefig(save_dir / 'depth_history.png')
            tmp_fig.clf()

        if params.obj_head != '':
            loss_history.plot(range(epochs), obj_loss_history)
            loss_history.plot(range(epochs), obj_val_loss_history)
            legend += ['obj','obj(val)']

            sub_fig = tmp_fig.add_subplot(1, 1, 1)
            sub_fig.plot(range(epochs), map50_history)
            tmp_fig.savefig(save_dir / 'map50_history.png')
            tmp_fig.clf()

            sub_fig = tmp_fig.add_subplot(1, 1, 1)
            sub_fig.plot(range(epochs), map5095_history)
            tmp_fig.savefig(save_dir / 'map5095_history.png')
            tmp_fig.clf()

        loss_history.legend(legend,loc='upper right')
        loss_history_fig.savefig(save_dir / 'loss_history.png')
        loss_history_fig.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',               type=str, help='root for Cityscapes', default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes')
    parser.add_argument('--project',            type=str, help='directory to save checkpoints and summaries', default='./runs/train/')
    parser.add_argument('--name',               type=str, help='save to project/name', default='mt')
    parser.add_argument('--encoder',            type=str, help='Choose Encoder in MT', default='densenet161')
    parser.add_argument('--weights',            type=str, help='initial weights path', default=None)
    parser.add_argument('--epochs',             type=int, help='number of epochs', default=100)
    parser.add_argument('--save-cycle',         type=int, help='save when cycle', default=90)
    parser.add_argument('--warmup',             type=int, help='epoch for warmpup', default=-1)
    parser.add_argument('--batch-size',         type=int, help='total batch size for all GPUs', default=8)
    parser.add_argument('--workers',            type=int, help='maximum number of dataloader workers', default=8)
    parser.add_argument('--input_height',       type=int,   help='input height', default=256)
    parser.add_argument('--input_width',        type=int,   help='input width',  default=512)
    parser.add_argument('--local_rank',         type=int,   help='DDP parameter, do not modify', default=-1)
    parser.add_argument('--max-cpu',            type=int,   help='Maximum CPU Usage(G) for Safety', default=20)
    parser.add_argument("--momentum",           type=float, help="Momentum component of the optimiser.", default=0.937)
    parser.add_argument('--weight_decay',       type=float, help='weight decay factor for optimization', default=1e-2)
    parser.add_argument('--learning_rate',      type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--end_learning_rate',  type=float, help='final OneCycleLR learning rate (lr0 * lrf)', default=1e-2)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--linear-learning-rate', action='store_true', help='linear Learning Rate or cosine curve')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plot', action='store_true', help='plot the loss and eval result')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # Augment
    parser.add_argument('--strategy',       action='store_true', help='use strategy to train model')
    parser.add_argument('--augment',        action='store_true', help='set for open augment')
    parser.add_argument('--random-hw',      type=float, default=0.0, help='random h and w in training')
    parser.add_argument('--random-flip',    type=float, default=0.5, help='flip the image and target')
    parser.add_argument('--random-crop',    type=float, default=0.5, help='crop the image and target')
    parser.add_argument('--multi-scale',    type=float, default=0.5, help='Image will be scaled proportionally')

    # Semantic Segmentation
    parser.add_argument('--semantic_head',      type=str, help='Choose method for semantic head(CCNet/HRNet/ESPNet)', default='CCNet')
    parser.add_argument('--smnt_num_classes',        type=int, help='Number of classes to predict (including background) for semantic segmentation.', default=19)

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)
    parser.add_argument('--depth_head',    type=str, help='Choose method for depth estimation head', default='bts')

    # Object detection
    parser.add_argument('--obj_head',      type=str, help='Choose method for obj detection head', default='yolo')
    parser.add_argument('--obj_num_classes',        type=int, help='Number of classes to predict (including background) for object detection.', default=80)
    params = parser.parse_args()

    # Directories
    params.save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok) # if create here will cause 2 folder
    params.save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("saving to " + str(params.save_dir))

    train(params)