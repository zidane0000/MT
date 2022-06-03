import cv2
import argparse
import torch
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path

from utils.general import increment_path, select_device, id2trainId, put_palette,LOGGER, reduce_tensor, safety_cpu, create_dataloader
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
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
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

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

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

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

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

def get_batch_statistics(outputs, targets, iou_thres):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
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
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


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
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def compute_yolo_eval(predicts, ground_truths, batch_metrics, conf_thres=0.25, iou_thres=0.45, nms_thres=0.4):
    """
    iou_thres  : IOU threshold required to qualify as detected
    conf_thres : Object confidence threshold
    nms_thres  : IOU threshold for non-maximum suppression
    """
    iouv = torch.linspace(0.5, 0.95, 10).to(ground_truths.device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    predicts = non_max_suppression(predicts, conf_thres=conf_thres, iou_thres=nms_thres)    
    
    # Metrics
    for si, pred in enumerate(predicts):        
        labels = ground_truths[ground_truths[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(pred) == 0:
            if nl:
                batch_metrics.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
            if plots:
                confusion_matrix.process_batch(predn, labelsn)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        batch_metrics.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
    
    return batch_metrics
            

def Concatenate_sample_statistics(sample_metrics):
    """
    sample_metrics : List of tuples (TP, confs, pred)
    """
    # Concatenate sample statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    true_positives, false_positives, percision, recall, f1, ap, ap_class = ap_per_class(*stats)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mean_percision, mean_recall, mean_ap50, meam_ap = percision.mean(), recall.mean(), ap50.mean(), ap.mean()
    return mean_percision, mean_recall, mean_ap50, meam_ap


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
    sample_metrics = []  # List of tuples (TP, confs, pred)

    for i, item in val_bar:
        img, smnt, depth, labels = item
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
                
        compute_yolo_eval(predict_obj, labels, sample_metrics)
        
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
    
    mean_percision, mean_recall, mean_ap50, meam_ap = Concatenate_sample_statistics(sample_metrics)

    LOGGER.info('%8s : %4.4f  ' % ('mean-IOU', smnt_mean_iou_val))
    depth_val_str = ['silog','abs_rel','log10','rmse','sq_rel','rmse_log','d1','d2','d3']
    for i in range(len(depth_val_str)):
        LOGGER.info('%8s : %5.3f' % (depth_val_str[i], depth_val[i]))
    
    LOGGER.info('mean_percision : %5.3f' % (mean_percision))
    LOGGER.info('mean_recall    : %5.3f' % (mean_recall))
    LOGGER.info('mean_ap50      : %5.3f' % (mean_ap50))
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