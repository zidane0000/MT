import cv2
import argparse
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path

from models.mt import MTmodel
from utils.loss import ComputeLoss
from utils.general import increment_path, select_device, id2trainId, put_palette
from utils.cityscapes import Create_Cityscapes

def val(params, save_dir=None, model=None, compute_loss=None):
    if save_dir is None:
        save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok, mkdir=True)
        print("saving to " + str(save_dir))

    if model is None:        
        device = select_device(params.device)
        print("begin load model with ckpt...")
        ckpt = torch.load(params.weight)
        model = MTmodel(params)
        if device != 'cpu' and torch.cuda.device_count() > 1:
            print("use multi-gpu, device=" + params.device)
            device_ids = [int(i) for i in params.device.split(',')]
            model = torch.nn.DataParallel(model, device_ids = device_ids)

        # if pt is save from multi-gpu, model need para first, see https://blog.csdn.net/qq_32998593/article/details/89343507
        model.load_state_dict(ckpt['model'])
        model.to(device)
        print(f"load model to device, from {params.weight}, epoch:{ckpt['epoch']}, train-time:{ckpt['date']}")        
    model.eval()

    # Dataset, DataLoader
    val_dataset, val_loader = Create_Cityscapes(params, mode='val')

    val_bar = enumerate(val_loader)
    val_bar = tqdm(val_bar, total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i, item in val_bar:
        img, (smnt, depth) = item
        img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        smnt = smnt.to(device)
        depth = depth.to(device)

        with torch.no_grad():
            output = model(img)

        if compute_loss:
            loss, (smnt_loss, depth_loss) = compute_loss(output, (smnt, depth))            

            # log
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            val_bar.set_description(('mem : %4s  ' + 'val-semantic : %4.4g  ' + 'val-depth : %4.4g') % (
                    mem, smnt_loss, depth_loss))
                
        if params.plot:
            (predict_smnt, predict_depth) = output
            interp = torch.nn.Upsample(size=(params.input_height, params.input_width), mode='bilinear', align_corners=True)
            predict_smnt = interp(predict_smnt)
            np_predict_smnt = predict_smnt.cpu().numpy().transpose(0,2,3,1)
            np_predict_smnt = np.asarray(np.argmax(np_predict_smnt, axis=3), dtype=np.uint8)
            np_predict_smnt = np_predict_smnt[0]
            np_predict_smnt = id2trainId(np_predict_smnt, reverse=True)
            np_predict_smnt = put_palette(np_predict_smnt, num_classes=255, path=str(save_dir) +'/smnt-' + str(i) + '.jpg')

            np_predict_depth = predict_depth[0].cpu().numpy().astype(np.float32).transpose(1,2,0)
            cv2.imwrite(str(save_dir) +'/depth-' + str(i) + '.jpg', np_predict_depth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',               type=str, default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes', help='root for Cityscapes')
    parser.add_argument('--project',            type=str, default='./runs/val/', help='directory to save checkpoints and summaries')
    parser.add_argument('--name',               type=str, default='mt', help='save to project/name')
    parser.add_argument('--weight',             type=str, default=None, help='model.pt path')
    parser.add_argument('--batch-size',         type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--workers',            type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--input_height',       type=int,   help='input height', default=256)
    parser.add_argument('--input_width',        type=int,   help='input width',  default=512)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plot', action='store_true', help='plot the loss and eval result')
    params = parser.parse_args()

    val(params)