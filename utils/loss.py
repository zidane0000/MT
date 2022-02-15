import torch
import torch.nn as nn
import torch.nn.functional as F


class silog_loss(nn.Module):
    '''
        scale-invariant error
        variance_focus -> higher enforce more focus on minimzing the variance of error
    '''
    def __init__(self, variance_focus=0.85):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt):
        # https://github.com/dg-enlens/banet-depth-prediction/blob/master/loss.py
        # let's only compute the loss on non-null pixels from the ground-truth depth-map
        print(depth_est.shape)
        print(depth_gt.shape)
        input()
        non_zero_mask = (depth_est > 0) & (depth_gt > 0)
        d = torch.log(depth_est[non_zero_mask]) - torch.log(depth_gt[non_zero_mask])
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
        '''
        preds : (batch_size, num_classes, height, width)
        target : (batch_size, height, width)
        '''
        h, w = target.size(1), target.size(2)

        target = target.to(dtype=torch.long)
        scale_pred = F.interpolate(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class ComputeLoss:
    def __init__(self):
        super(ComputeLoss, self).__init__()
        self.semantic_loss_function = CriterionDSN()
        self.depth_loss_function = silog_loss()
    
    def __call__(self, predicts, targets):
        (predict_smnt, predict_depth) = predicts
        (target_smnt, target_depth) = targets

        smnt_loss = self.semantic_loss_function(predict_smnt, target_smnt)
        depth_loss = self.depth_loss_function(predict_depth, target_depth)

        return (smnt_loss + depth_loss), (smnt_loss, depth_loss)


