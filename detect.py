import os
import cv2
import argparse
import torch
import numpy as np
import time
import torchvision
import glob
import matplotlib.pyplot as plt

from pathlib import Path

from utils.general import increment_path, select_device, id2trainId, put_palette,LOGGER, reduce_tensor, safety_cpu, create_dataloader, xywh2xyxy, xyxy2xywh, plot_xywh
from models.mt import MTmodel

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


class LoadImages:
    def __init__(self, path):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img = self.cap.read()

            self.frame += 1
            log = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
            assert img is not None, f'Image Not Found {path}'
            log = f'image {self.count}/{self.nf} {path}: '

        # Convert
        img = np.moveaxis(img, -1, 0)
        img = np.ascontiguousarray(img)

        return path, img, self.cap, log

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def detect(params, save_dir=None, model=None, device=None):
    if save_dir is None:
        save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok, mkdir=True)
        LOGGER.info("saving to " + str(save_dir))

    device = select_device(params.device)
    LOGGER.info("begin load model with ckpt...")
    ckpt = torch.load(params.weight)
    model = MTmodel(params)
    if device != 'cpu' and torch.cuda.device_count() > 1:
        LOGGER.info("use multi-gpu, device=" + params.device)
        device_ids = [int(i) for i in params.device.split(',')]
        model = torch.nn.DataParallel(model, device_ids = device_ids)

    # if pt is save from multi-gpu, model need para first, see https://blog.csdn.net/qq_32998593/article/details/89343507
    model.load_state_dict(ckpt['model'])
    model.to(device)
    LOGGER.info(f"load model to device, from {params.weight}, epoch:{ckpt['epoch']}, train-time:{ckpt['date']}")
    model.eval()

    dataset = LoadImages(params.path)
    
    save_video = params.save_video
    if len(dataset) > 1 and save_video:
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 影片編碼格式
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        out_video_path = str(save_dir) + '/detect.mp4'
        height, width = 1024, 2048
        cap_out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    else:
        save_video = False
    
    count = 0
    for path, img, cap, log in dataset:
        t1 = time_sync()
        img = torch.Tensor(img).to(device)
        img /= 255  # 0 - 255 to 0.0 - 1.0
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()

        # Inference
        with torch.no_grad():
            output = model(img)
        t3 = time_sync()

        np_img = (img[0]).cpu().numpy().transpose(1,2,0)
        np_img *= 255
        np_img = np_img.astype(np.uint8)
        
        np_predict_smnt = np.zeros_like(np_img)
        np_predict_depth = np.zeros_like(np_img)
        np_predict_obj = np.zeros_like(np_img)

        task = 0
        t4 = time_sync()
        if params.semantic_head != '':
            predict_smnt = output[task]
            task+=1

            np_predict_smnt = predict_smnt.cpu().numpy()
            np_predict_smnt = np.asarray(np.argmax(np_predict_smnt, axis=1), dtype=np.uint8) # batch, class, w, h -> batch, w, h
            np_predict_smnt = id2trainId(np_predict_smnt[0], 255, reverse=True)
            np_predict_smnt = put_palette(np_predict_smnt, num_classes=255)

        t5 = time_sync()   
        if params.depth_head != '':
            predict_depth = output[task]
            task+=1

            np_predict_depth = predict_depth[0].cpu().numpy().squeeze().astype(np.float32)

        t6 = time_sync()
        if params.obj_head != '':
            (predict_obj, predict_train_obj) = output[task]
            task+=1

            conf_thres = 0.001
            iou_thres = 0.6
            single_cls = False
            out = predict_obj
            out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True, agnostic=single_cls)
            targets = []
            for oi, o in enumerate(out):
                for *box, conf, cls in o.cpu().numpy():
                    targets.append([oi, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
            targets = np.array(targets)
            targets = targets[targets[:,0]==0]
            targets = targets[targets[:,6]>0.25]
            np_predict_obj = plot_xywh(np_img, targets[:,1:6])
        t7 = time_sync()

        if save_video:
            each_width, each_height = width // 2, height // 2
            np_img = cv2.resize(np_img, (each_width, each_height), interpolation=cv2.INTER_LINEAR)
            np_predict_smnt = cv2.resize(np_predict_smnt, (each_width, each_height), interpolation=cv2.INTER_NEAREST)
            np_predict_depth = cv2.resize(np_predict_depth, (each_width, each_height), interpolation=cv2.INTER_NEAREST)
            np_predict_obj = cv2.resize(np_predict_obj, (each_width, each_height), interpolation=cv2.INTER_LINEAR)

            cv2.putText(np_predict_smnt, 'Semantic segmetation', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(np_predict_depth, 'Depth estimation', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(np_predict_obj, 'Object detection', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)
            
            np_predict_depth = cv2.cvtColor(np_predict_depth,cv2.COLOR_GRAY2BGR).astype(np.uint8)
            img_top = cv2.hconcat([np_img, np_predict_obj])
            img_bottom = cv2.hconcat([np_predict_depth, np_predict_smnt])
            img_concat = cv2.vconcat([img_top, img_bottom])
            cap_out.write(img_concat)
        else:
            ori_name = log[log.rfind('/')+1:log.find('.png')]
            cv2.imwrite(str(save_dir) +'/'+ ori_name +'_img.jpg', np_img)
            if params.semantic_head != '':
                cv2.imwrite(str(save_dir) +'/'+ ori_name +'_smnt.jpg', np_predict_smnt)
            if params.depth_head != '':
                cv2.imwrite(str(save_dir) +'/'+ ori_name +'_depth.jpg', np_predict_depth)
            if params.obj_head != '':
                cv2.imwrite(str(save_dir) +'/'+ ori_name +'_obj.jpg', np_predict_obj)
            
        LOGGER.info(f'{log}Done. (inference+{t3 - t2:.3f}s smnt+{t5 - t4:.3f}s depth+{t6 - t5:.3f}s obj+{t7 - t6:.3f}s)')
            

    if save_video:
        cap_out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',               type=str, default=None, help='Folder path or video path for detect')
    parser.add_argument('--project',            type=str, default='./runs/detect/', help='directory to save checkpoints and summaries')
    parser.add_argument('--name',               type=str, default='mt', help='save to project/name')
    parser.add_argument('--encoder',            type=str, default='densenet161', help='Choose Encoder in MT')
    parser.add_argument('--weight',             type=str, default=None, help='model.pt path')
    parser.add_argument('--device',             default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--exist-ok',           action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-video',         action='store_true', help='save result(concat) as video')   

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

    detect(params=params)
