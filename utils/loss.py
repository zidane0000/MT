import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as nd
import numpy as np


def scale_invariant_loss(predicts: torch.Tensor, targets: torch.Tensor, reduction="mean"):
    """
    outs: N ( x C) x H x W
    targets: N ( x C) x H x W
    reduction: ...
    """
    predicts = predicts.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    alpha = (targets - outs).mean(dim=1, keepdim=True)
    return F.mse_loss(outs + alpha, targets, reduction=reduction)


class silog_loss(nn.Module):
    '''
        scale-invariant error
        variance_focus -> higher enforce more focus on minimzing the variance of error
    '''
    def __init__(self, variance_focus=0.85):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, predict_depth, target_depth):
        if predict_depth.dim() > target_depth.dim():
            # (batch, h, w) -> (h, w)
            target_depth = torch.unsqueeze(target_depth, 1)
        if predict_depth.dtype != target_depth.dtype:
            predict_depth = predict_depth.float()
            target_depth = target_depth.float()
        # https://github.com/dg-enlens/banet-depth-prediction/blob/master/loss.py
        # let's only compute the loss on non-null pixels from the ground-truth depth-map
        non_zero_mask = (predict_depth > 0) & (target_depth > 0)
        d = torch.log(predict_depth[non_zero_mask]) - torch.log(target_depth[non_zero_mask])
        # scaling the range of loss improve convergence and final result, and 10.0 is constant
        constant = 10.0
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * constant

    
class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduction='mean'):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        # 交叉熵計算Loss，忽略了255類，並且對Loss取了平均
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        if not reduction:
            print("disabled the reduction.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        ph, pw = preds.size(2), preds.size(3)

        if ph != h or pw != w:
            preds = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
                    
        if target.dtype != torch.long:
            target = target.to(torch.long)
        loss = self.criterion(preds, target)
        return loss

    
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        # self.min_kept_ratio = float(min_kept_ratio)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    """ 閾值的選取主要基於min_kept，用第min_kept個的概率來確定。且返回的閾值只能≥thresh。 """
    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0/factor, 1.0/factor), order=1) # 雙線性插值
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0) # 最近臨插值

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor*factor) #int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32) # 將多維數組轉化為一維
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept)-1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    """ 
        主要思路 
            1.先通過find_threshold找到一個合適的閾值如0.7 
            2.一次篩選出不為255的區域 
            3.再從中二次篩選找出對應預測值小於0.7的區域 
            4.重新生成一個label，label把預測值大於0.7和原本為255的位置 都置為255
    """
    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target

    def forward(self, predict, target, weight=None):        
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)        
        
        return self.criterion(predict, target)


class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        ph, pw = preds.size(2), preds.size(3)

        if ph != h or pw != w:
            preds = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
                        
        loss1 = self.criterion1(preds, target)
        
        if target.dtype != torch.long:
            target = target.to(torch.long)
        loss2 = self.criterion2(preds, target)

        return loss1 + loss2*0.4


class ComputeLoss:
    def __init__(self):
        super(ComputeLoss, self).__init__()
        self.semantic_loss_function = CriterionDSN() # CriterionOhemDSN()
        self.depth_loss_function = silog_loss()
    
    def __call__(self, predicts, targets):
        (predict_smnt, predict_depth) = predicts
        (target_smnt, target_depth) = targets

        depth_loss = self.depth_loss_function(predict_depth, target_depth)
        depth_loss = torch.unsqueeze(depth_loss, 0) # 0 dim to 1 dim, like 10 -> [10]
        
        smnt_loss = self.semantic_loss_function(predict_smnt, target_smnt)
        smnt_loss = torch.unsqueeze(smnt_loss, 0) # 0 dim to 1 dim, like 10 -> [10]

        return (smnt_loss + depth_loss), (smnt_loss, depth_loss)
