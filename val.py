import cv2
import argparse
import torch
import numpy as np
import time
import torchvision
from tqdm import tqdm
from pathlib import Path

from utils.general import increment_path, select_device, id2trainId, put_palette,LOGGER, reduce_tensor, safety_cpu, create_dataloader, xywh2xyxy
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


# Inherit from yolo
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

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
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        output = output.to(targets.device)
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))

                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_targets)).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
    return batch_metrics


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_yolo_eval(predicts, ground_truths, conf_thres=0.25, nms_thres=0.4, iou_thres=0.5):
    """
    conf_thres    : Object confidence threshold
    nms_thres     : IOU threshold for non-maximum suppression
    iou_thres     : IOU threshold required to qualify as detected
    """
    iouv = torch.linspace(0.5, 0.95, 10).to(ground_truths.device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    predicts = non_max_suppression(predicts, conf_thres=conf_thres, iou_thres=nms_thres)
    batch_metrics = get_batch_statistics(predicts, ground_truths, iou_threshold=iou_thres)    
    return batch_metrics
            

def Concatenate_sample_statistics(sample_metrics, all_labels):
    """
    sample_metrics : List of tuples (TP, confs, pred)
    """
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    percision, recall, ap, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, all_labels)
    mean_percision, mean_recall = percision.mean(), recall.mean()
    meam_ap = ap.mean()
    return mean_percision, mean_recall, meam_ap


def val(params, save_dir=None, model=None, device=None, compute_loss=None, val_loader=None):
    if save_dir is None:
        save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok, mkdir=True)
        LOGGER.info("saving to " + str(save_dir))

    if model is None:
        from models.mt import MTmodel
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
    mean_loss = torch.zeros(3, device=device)
    smnt_mean_iou_val = 0
    smnt_iou_array_val = np.zeros((params.num_classes,params.num_classes))
    depth_val = np.zeros(9)
    
    all_labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    for i, item in val_bar:
        img, smnt, depth, labels = item
        all_labels += labels[:, 1].tolist()
        img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        smnt = smnt.to(device)
        depth = depth.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(img)
        
        (predict_smnt, predict_depth, (predict_obj, predict_train_obj)) = output

        if compute_loss:
            safety_cpu(params.max_cpu)
            loss, (smnt_loss, depth_loss, obj_loss) = compute_loss((predict_smnt, predict_depth, predict_train_obj), (smnt, depth, labels))
            mem = f'{torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            mean_loss = (mean_loss * i + torch.cat((smnt_loss, depth_loss, obj_loss)).detach()) / (i + 1)
            val_bar.set_description((' '*16 + 'mem:%8s' + '  val-semantic:%6.6g' + '  val-depth:%6.6g' + '  val-obj:%6.6g') % (
                                        mem, mean_loss[0], mean_loss[1], mean_loss[2]))
                
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
                
        sample_metrics += compute_yolo_eval(predict_obj, labels)
        
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
    
    mean_percision, mean_recall, meam_ap = Concatenate_sample_statistics(sample_metrics, all_labels)

    LOGGER.info('%8s : %4.4f  ' % ('mean-IOU', smnt_mean_iou_val))
    depth_val_str = ['silog','abs_rel','log10','rmse','sq_rel','rmse_log','d1','d2','d3']
    for i in range(len(depth_val_str)):
        LOGGER.info('%8s : %5.3f' % (depth_val_str[i], depth_val[i]))
    
    LOGGER.info('mean_percision : %5.3f' % (mean_percision))
    LOGGER.info('mean_recall    : %5.3f' % (mean_recall))
    LOGGER.info('meam_ap        : %5.3f' % (meam_ap))
        
    LOGGER.info('-'*45)

    if compute_loss:
        return (mean_loss[0], mean_loss[1]), (smnt_mean_iou_val, smnt_iou_array_val), depth_val
    else:
        return (smnt_mean_iou_val, smnt_iou_array_val), depth_val
    

def val_one(params, save_dir=None, model_type=None, model=None, device=None, compute_loss=None, val_loader=None, task=None):    
    if save_dir is None:
        save_dir = increment_path(Path(params.project) / params.name, exist_ok=params.exist_ok, mkdir=True)
        LOGGER.info("saving to " + str(save_dir))

    if model is None:
        if model_type in ['ccnet','espnet', 'hrnet']:  
            task = 'smnt'
            if model_type == 'espnet':
                from models.decoder.espnet import ESPNet as OneModel
            elif model_type == 'hrnet':
                from models.decoder.hrnet_ocr import HighResolutionNet as OneModel, cfg as hrnet_cfg
        elif model_type.lower() in ['bts','yolor']:  
            task = 'depth'
            if model_type == 'bts':
                from models.decoder.bts import BtsModel as OneModel
            elif model_type == 'yolor':
                from models.yolo import YOLOR_depth as OneModel
        assert OneModel is not None, 'Unkown OneModel'
        device = select_device(params.device)
        LOGGER.info("begin load model with ckpt...")
        ckpt = torch.load(params.weight)
        cfg = hrnet_cfg if model_type == 'hrnet' else params
        model = OneModel(cfg)
        if device != 'cpu' and torch.cuda.device_count() > 1:
            LOGGER.info("use multi-gpu, device=" + params.device)
            device_ids = [int(i) for i in params.device.split(',')]
            model = torch.nn.DataParallel(model, device_ids = device_ids)

        # if pt is save from multi-gpu, model need para first, see https://blog.csdn.net/qq_32998593/article/details/89343507
        model.load_state_dict(ckpt['model'])
        model.to(device)
        LOGGER.info(f"load model to device, from {params.weight}, epoch:{ckpt['epoch']}, train-time:{ckpt['date']}")
    model.eval()    
    
    if task is None:
        LOGGER.info("No define Task")
        return None

    # Dataset, DataLoader
    if val_loader == None:
        val_dataset, val_loader = create_dataloader(params, mode='val')

    val_bar = enumerate(val_loader)
    val_bar = tqdm(val_bar, total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    
    # val result
    mean_loss = torch.zeros(1, device=device)
    smnt_mean_iou_val = 0
    smnt_iou_array_val = np.zeros((params.num_classes,params.num_classes))
    depth_val = np.zeros(9)

    for i, item in val_bar:
        img, smnt, depth, labels = item
        img = img.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        
        if task == "depth":
            gt = depth.to(device)
        elif task == "smnt":
            gt = smnt.to(device)
        else:
            gt = None

        with torch.no_grad():
            output = model(img)
            output = output[-1] if model_type == 'yolor' else output

        if compute_loss:
            safety_cpu(params.max_cpu)
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
            
            if params.plot:
                np_gt_smnt = np_gt_smnt[0]
                np_gt_smnt = id2trainId(np_gt_smnt, 255, reverse=True)
                np_gt_smnt = put_palette(np_gt_smnt, num_classes=255, path=str(save_dir) +'/smnt-gt-' + str(i) + '.jpg')
                
                np_predict_smnt = np_predict_smnt[0]
                np_predict_smnt = id2trainId(np_predict_smnt, 255, reverse=True)
                np_predict_smnt = put_palette(np_predict_smnt, num_classes=255, path=str(save_dir) +'/smnt-' + str(i) + '.jpg')
        elif task == "depth":
            np_predict_depth = output.cpu().numpy().squeeze().astype(np.float32)
            np_gt_depth = depth.cpu().numpy().astype(np.float32)
            depth_val += np.array(compute_bts_eval(np_predict_depth, np_gt_depth, params.min_depth, params.max_depth))
            
            if params.plot:
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

        if params.plot:
            np_img = (img[0] * 255).cpu().numpy().astype(np.int64).transpose(1,2,0)
            cv2.imwrite(str(save_dir) +'/img-' + str(i) + '.jpg', np_img)
    
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
    parser.add_argument('--max-cpu',            type=int, default=20,  help='Maximum CPU Usage(G) for Safety')
    parser.add_argument('--device',             default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--exist-ok',           action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plot',               action='store_true', help='plot the loss and eval result')
    parser.add_argument('--random-flip',        action='store_true', help='flip the image and target')
    parser.add_argument('--random-crop',        action='store_true', help='crop the image and target')
    
    # MT or One
    parser.add_argument('--model_type',         type=str, default='mt', help='Choose Model Type by lower, mt or one model')
    
    # Semantic Segmentation
    parser.add_argument('--num_classes',        type=int, help='Number of classes to predict (including background).', default=19)
    parser.add_argument('--semantic_head',      type=str, help='Choose method for semantic head(CCNet/HRNet/ESPNet)', default='CCNet')

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)
    parser.add_argument('--depth_head',    type=str, help='Choose method for depth estimation head', default='bts') 
    
    # Object detection
    parser.add_argument('--obj_head',      type=str, help='Choose method for obj detection head', default='yolo')
    params = parser.parse_args()
    
    params.model_type = params.model_type.lower()
    if params.model_type == 'mt':
        val(params=params)
    else:
        val_one(params=params,model_type=params.model_type)