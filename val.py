import cv2
import argparse
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path

from models.mt import MTmodel
from utils.loss import ComputeLoss
from utils.general import increment_path, select_device, id2trainId, put_palette,LOGGER, reduce_tensor, safety_cpu
from utils.cityscapes import Create_Cityscapes


def compute_ccnet_eval(predicts, ground_truths, class_num):
    def get_confusion_matrix(pred_label, gt_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param pred_label: the pred label
        :param gt_label: the ground truth label        
        :param class_num: the numnber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))
        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
        return confusion_matrix
    
    confusion_matrix = np.zeros((class_num, class_num))
    for i in range(len(predicts)):
        ignore_index = ground_truths[i] != 255
        predict_mask = predicts[i][ignore_index]
        ground_truth_mask = ground_truths[i][ignore_index]
        confusion_matrix += get_confusion_matrix(predict_mask, ground_truth_mask, class_num)
    
    if torch.distributed.is_initialized():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    return mean_IoU, IoU_array


def compute_bts_eval(predicts, ground_truths, min_depth, max_depth):
    silog, log10, rmse, rmse_log, abs_rel, sq_rel, d1, d2, d3 = (np.zeros(len(predicts), np.float32) for i in range(9))

    for i in range(len(predicts)):
        predicts[i][predicts[i] < min_depth] = min_depth
        predicts[i][predicts[i] > max_depth] = max_depth
        predicts[i][np.isinf(predicts[i])] = max_depth
        predicts[i][np.isnan(predicts[i])] = min_depth

        valid_mask = np.logical_and(ground_truths[i] > min_depth, ground_truths[i] < max_depth)
        predict = predicts[i][valid_mask]        
        ground_truth = ground_truths[i][valid_mask]

        thresh = np.maximum((ground_truth / predict), (predict / ground_truth))
        d1[i] = (thresh < 1.25).mean()
        d2[i] = (thresh < 1.25 ** 2).mean()
        d3[i] = (thresh < 1.25 ** 3).mean()
        
        tmp = (ground_truth - predict) ** 2
        rmse[i] = np.sqrt(tmp.mean())
        
        tmp = (np.log(ground_truth) - np.log(predict)) ** 2
        rmse_log[i] = np.sqrt(tmp.mean())
        
        abs_rel[i] = np.mean(np.abs(ground_truth - predict) / ground_truth)
        sq_rel[i] = np.mean(((ground_truth - predict) ** 2) / ground_truth)
        
        err = np.log(predict) - np.log(ground_truth)
        silog[i] = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
        
        err = np.abs(np.log10(predict) - np.log10(ground_truth))
        log10[i] = np.mean(err)
    
    return silog.mean(), abs_rel.mean(), log10.mean(), rmse.mean(), sq_rel.mean(), rmse_log.mean(), d1.mean(), d2.mean(), d3.mean()


def val(params, save_dir=None, model=None, device=None, compute_loss=None, val_loader=None):
    if save_dir is None:
        save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok, mkdir=True)
        LOGGER.info("saving to " + str(save_dir))

    if model is None:        
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

    # Dataset, DataLoader
    if val_loader == None:
        val_dataset, val_loader = Create_Cityscapes(params, mode='val')

    val_bar = enumerate(val_loader)
    val_bar = tqdm(val_bar, total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    # val result
    mean_loss = torch.zeros(2, device=device)
    smnt_mean_iou_val = 0
    smnt_iou_array_val = np.zeros((params.num_classes,params.num_classes))
    depth_val = np.zeros(9)

    for i, item in val_bar:
        img, (smnt, depth) = item
        img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        smnt = smnt.to(device)
        depth = depth.to(device)

        with torch.no_grad():
            output = model(img)

        if compute_loss:
            safety_cpu()
            loss, (smnt_loss, depth_loss) = compute_loss(output, (smnt, depth))
            mem = f'{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            mean_loss = (mean_loss * i + torch.cat((smnt_loss, depth_loss)).detach()) / (i + 1)
            val_bar.set_description((' '*16 + 'mem:%8s' + '  val-semantic:%6.6g' + '  val-depth:%6.6g') % (
                                        mem, mean_loss[0], mean_loss[1]))
        
        (predict_smnt, predict_depth) = output
        # upsample to origin size
        interp = torch.nn.Upsample(size=(params.input_height, params.input_width), mode='bilinear', align_corners=True)
        predict_smnt = interp(predict_smnt)

        np_predict_smnt = predict_smnt.cpu().numpy()
        np_predict_smnt = np.asarray(np.argmax(np_predict_smnt, axis=1), dtype=np.uint8) # batch, class, w, h -> batch, w, h
        np_gt_smnt= smnt.cpu().numpy()
        miou, iou_array = compute_ccnet_eval(np_predict_smnt, np_gt_smnt, params.num_classes)
        smnt_mean_iou_val += miou
        smnt_iou_array_val += iou_array

        np_predict_depth = predict_depth.cpu().numpy().squeeze().astype(np.float32)
        np_gt_depth = depth.cpu().numpy().astype(np.float32)
        depth_val += np.array(compute_bts_eval(np_predict_depth, np_gt_depth, params.min_depth, params.max_depth))
                
        if params.plot:
            np_gt_smnt = np_gt_smnt[0]
            np_gt_smnt = id2trainId(np_gt_smnt, 255, reverse=True)
            np_gt_smnt = put_palette(np_gt_smnt, num_classes=255, path=str(save_dir) +'/smnt-gt-' + str(i) + '.jpg')
            
            np_predict_smnt = np_predict_smnt[0]
            np_predict_smnt = id2trainId(np_predict_smnt, 255, reverse=True)
            np_predict_smnt = put_palette(np_predict_smnt, num_classes=255, path=str(save_dir) +'/smnt-' + str(i) + '.jpg')

            np_predict_depth = np_predict_depth[0]
            cv2.imwrite(str(save_dir) +'/depth-' + str(i) + '.jpg', np_predict_depth)
            
            heat_predict_depth = (np_predict_depth * 255).astype('uint8')
            heat_predict_depth = cv2.applyColorMap(heat_predict_depth, cv2.COLORMAP_JET)
            cv2.imwrite(str(save_dir) +'/heat-' + str(i) + '.jpg', heat_predict_depth)
            
            np_gt_depth = np_gt_depth[0]
            cv2.imwrite(str(save_dir) +'/depth-gt-' + str(i) + '.jpg', np_gt_depth)
            
            heat_gt_depth = (np_gt_depth * 255).astype('uint8')
            heat_gt_depth = cv2.applyColorMap(heat_gt_depth, cv2.COLORMAP_JET)
            cv2.imwrite(str(save_dir) +'/heat-gt-' + str(i) + '.jpg', heat_gt_depth)

            np_img = (img[0] * 255).cpu().numpy().astype(np.int64).transpose(1,2,0)
            cv2.imwrite(str(save_dir) +'/img-' + str(i) + '.jpg', np_img)
    
    smnt_mean_iou_val /= len(val_bar)
    smnt_iou_array_val /= len(val_bar)
    depth_val /= len(val_bar)

    LOGGER.info('%8s : %4.4f  ' % ('mean-IOU', smnt_mean_iou_val))
    depth_val_str = ['silog','abs_rel','log10','rmse','sq_rel','rmse_log','d1','d2','d3']
    for i in range(len(depth_val_str)):
        LOGGER.info('%8s : %5.3f' % (depth_val_str[i], depth_val[i]))
    LOGGER.info('-'*45)

    if compute_loss:
        return (mean_loss[0], mean_loss[1]), (smnt_mean_iou_val, smnt_iou_array_val), depth_val
    else:
        return (smnt_mean_iou_val, smnt_iou_array_val), depth_val
    

def val_one(params, save_dir=None, model=None, device=None, compute_loss=None, val_loader=None, task=None):
    if task is None:
        LOGGER.info("No define Task")
        return None
    
    if save_dir is None:
        save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok, mkdir=True)
        LOGGER.info("saving to " + str(save_dir))

    if model is None:        
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

    # Dataset, DataLoader
    if val_loader == None:
        val_dataset, val_loader = Create_Cityscapes(params, mode='val')

    val_bar = enumerate(val_loader)
    val_bar = tqdm(val_bar, total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    # val result
    mean_loss = torch.zeros(1, device=device)
    smnt_mean_iou_val = 0
    smnt_iou_array_val = np.zeros((params.num_classes,params.num_classes))
    depth_val = np.zeros(9)

    for i, item in val_bar:
        img, (smnt, depth) = item
        img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        
        if task == "depth":
            gt = depth.to(device)
        elif task == "smnt":
            gt = smnt.to(device)
        else:
            gt = None

        with torch.no_grad():
            output = model(img)

        if compute_loss:
            safety_cpu()
            loss = compute_loss(output, gt)
            mem = f'{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            mean_loss = (mean_loss * i + loss) / (i + 1)
            val_bar.set_description((' '*16 + 'mem:%8s' + '  loss:%6.6g') % (mem, mean_loss))

        # upsample to origin size
        if task == "smnt":
            interp = torch.nn.Upsample(size=(params.input_height, params.input_width), mode='bilinear', align_corners=True)
            predict_smnt = interp(output)

            np_predict_smnt = predict_smnt.cpu().numpy()
            np_predict_smnt = np.asarray(np.argmax(np_predict_smnt, axis=1), dtype=np.uint8) # batch, class, w, h -> batch, w, h
            np_gt_smnt= smnt.cpu().numpy()
            miou, iou_array = compute_ccnet_eval(np_predict_smnt, np_gt_smnt, params.num_classes)
            smnt_mean_iou_val += miou
            smnt_iou_array_val += iou_array            
        elif task == "depth":
            np_predict_depth = output.cpu().numpy().squeeze().astype(np.float32)
            np_gt_depth = depth.cpu().numpy().astype(np.float32)
            depth_val += np.array(compute_bts_eval(np_predict_depth, np_gt_depth, params.min_depth, params.max_depth))
    
    smnt_mean_iou_val /= len(val_bar)
    smnt_iou_array_val /= len(val_bar)
    depth_val /= len(val_bar)
    
    if task== "smnt":
        LOGGER.info('%8s : %4.4f  ' % ('mean-IOU', smnt_mean_iou_val))
    elif task == "depth":
        depth_val_str = ['silog','abs_rel','log10','rmse','sq_rel','rmse_log','d1','d2','d3']
        for i in range(len(depth_val_str)):
            LOGGER.info('%8s : %5.3f' % (depth_val_str[i], depth_val[i]))
    LOGGER.info('-'*45)

    if compute_loss:
        return mean_loss, (smnt_mean_iou_val, smnt_iou_array_val), depth_val
    else:
        return (smnt_mean_iou_val, smnt_iou_array_val), depth_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',               type=str, default='/home/user/hdd2/Autonomous_driving/datasets/cityscapes', help='root for Cityscapes')
    parser.add_argument('--project',            type=str, default='./runs/val/', help='directory to save checkpoints and summaries')
    parser.add_argument('--name',               type=str, default='mt', help='save to project/name')
    parser.add_argument('--encoder',            type=str, default='densenet161', help='Choose Encoder in MT')
    parser.add_argument('--weight',             type=str, default=None, help='model.pt path')
    parser.add_argument('--batch-size',         type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--workers',            type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--input_height',       type=int, default=256, help='input height')
    parser.add_argument('--input_width',        type=int, default=512, help='input width')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plot', action='store_true', help='plot the loss and eval result')
    parser.add_argument('--random-flip', action='store_true', help='flip the image and target')
    parser.add_argument('--random-crop', action='store_true', help='crop the image and target')
    # Semantic Segmentation
    parser.add_argument('--num_classes',            type=int, help='Number of classes to predict (including background).', default=19)

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)
    params = parser.parse_args()

    val(params)