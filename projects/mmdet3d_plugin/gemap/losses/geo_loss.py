import torch
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from mmdet.models.builder import LOSSES
from mmdet.models import weighted_loss
import mmcv
import torch.nn.functional as F
from mmdet.core.bbox.match_costs.builder import MATCH_COST
import functools


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


@LOSSES.register_module()
class GeometricLoss(nn.Module):
    """
        Implementation of Geometric Loss
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        intra_loss_weight (float, optional): The weight of Euclidean shape loss.
        inter_loss_weight (float, optional): The weight of Euclidean relation loss.
        num_ins (int, optional): The number of instances.
        num_pts (int, optional): The number of fixed points of each instance.
            --------------------------------------------------
            The indices of predictions is organized as follows:

            Instance 0: {0, 1, ..., num_pts-1}
            Instance 1: {num_pts, num_pts+1, ..., 2*num_pts-1}
            ...
            Instance num_ins-1: {(num_ins-1)*num_pts, (num_ins-1)*num_pts+1, ..., num_ins*num_pts-1}
            ---------------------------------------------------
        num_classes (int, optional): The number of instance categories
            "num_classes + 1" is adopted to mark prediction which is matched to no gt.
        pc_range (list[float], optional): The range of lidar point clouds, formated as follows:
            [x_min, y_min, z_min, x_max, y_max, z_max]
        loss_type (str, optional): The type of loss to measure dicrepancies between preds and gt.
            Options are "l1".   
    """

    def __init__(
            self, reduction='mean', loss_weight=1.0, 
            intra_loss_weight=1.0, inter_loss_weight=1.0,
            num_ins=50, num_pts=20, 
            num_classes=3,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            loss_type='l1',
        ):
        super(GeometricLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.inter_loss_weight = inter_loss_weight
        self.num_ins = num_ins
        self.num_pts = num_pts
        self.num_classes = num_classes
        self.pc_range = pc_range
        if loss_type == 'l1':
            self.loss_geo = l1_loss
        else:
            raise NotImplementedError('Only "l1" is supported as "loss_type".')
    
    @staticmethod
    def batch_cross_product(a, b):
        ax, ay = a[:, :, 0], a[:, :, 1]
        bx, by = b[:, :, 0], b[:, :, 1]
        return torch.mul(ax, by) - torch.mul(bx, ay)
    
    @staticmethod
    def batch_dot_product(a, b):
        return torch.sum(torch.mul(a, b), dim=-1)
    
    def compute_intra_geometrics(self, x):
        offset_x = x - torch.roll(x, shifts=1, dims=1)
        # intra offset length
        length = torch.norm(offset_x, p=2, dim=-1).flatten()
        # intra offset theta
        rl_offset_x = torch.roll(offset_x, shifts=1, dims=1)
        rl_offset_x_cross = torch.stack([
            rl_offset_x[:, :, 1],
            -rl_offset_x[:, :, 0]
        ], dim=-1)
        norms = (torch.norm(offset_x, p=2, dim=-1) * torch.norm(rl_offset_x, p=2, dim=-1))
        dots = torch.matmul(offset_x.unsqueeze(2), rl_offset_x.unsqueeze(3))
        dots = dots.squeeze(-1).squeeze(-1)/(norms + 1e-6)
        dots = dots.flatten()
        crosses = torch.matmul(offset_x.unsqueeze(2), rl_offset_x_cross.unsqueeze(3))
        crosses = crosses.squeeze(-1).squeeze(-1)/(norms + 1e-6)
        crosses = crosses.flatten()
        return length, dots, crosses
    
    def compute_inter_geometrics(self, x, offset_x):
        N = x.shape[0]  # the number of instances
        inter_mask = torch.ones((N, N), device=x.device)
        for i in range(N):
            inter_mask[i, i:] = 0  # avoid redundant computation
        inter_mask = inter_mask.flatten()
        x_src = x.repeat((N, 1, 1))
        x_tgt = torch.cat([
            x[i].unsqueeze(0).repeat((N, 1, 1)) for i in range(N)
        ], dim=0)
        length = torch.norm(x_src.unsqueeze(2) - x_tgt.unsqueeze(1), p=2, dim=-1) \
                    * inter_mask.unsqueeze(-1).unsqueeze(-1) # for each pair of instances, compute point-to-point offset length
        offset_x_src = offset_x.repeat((N, 1, 1))
        offset_x_tgt = torch.cat([
            offset_x[i].unsqueeze(0).repeat((N, 1, 1)) for i in range(N)
        ], dim=0)
        offset_x_tgt_cross = torch.stack([
            offset_x_tgt[:, :, 1],
            -offset_x_tgt[:, :, 0]
        ], dim=-1)
        norms = torch.matmul(torch.norm(offset_x_src, p=2, dim=-1).unsqueeze(2), torch.norm(offset_x_tgt, p=2, dim=-1).unsqueeze(1))
        dots = torch.matmul(offset_x_src, offset_x_tgt.transpose(-1, -2)) * inter_mask.unsqueeze(-1).unsqueeze(-1)
        dots = dots / (norms + 1e-6)
        dots = dots.flatten()
        crosses = torch.matmul(offset_x_src, offset_x_tgt_cross.transpose(-1, -2)) * inter_mask.unsqueeze(-1).unsqueeze(-1)
        crosses = crosses / (norms + 1e-6)
        crosses = crosses.flatten()
        return length, dots, crosses

    def forward(self,
                pred,
                target,
                labels,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """
            Forward function.
        Args:
            pred (torch.Tensor): The prediction of shape (B * N, Nv, 2)
                ----------------------------------------------
                B: batch size
                N: the number of instances of each sample
                Nv: the number points on each instance 
                ----------------------------------------------
            target (torch.Tensor): The learning target of the prediction of shape (B * N, Nv, 2)
            labels (torch.Tensor): The predicted labels of each instance
                ----------------------------------------------
                "self.num_classes + 1" is adopted to mark the prediction which is matched to no gt.
                The geometric loss requires this information to avoid contribution from unmatched 
                instances, especially in Euclidean relation loss.
                ----------------------------------------------
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        normalized_target = normalize_2d_pts(target, self.pc_range)
        # normalize target points to [-1, 1]

        # intra, shape
        intra_loss = 0
        ft_preds = pred[labels < self.num_classes]
        ft_targets = normalized_target[labels < self.num_classes]
        length_preds, dot_preds, cross_preds = self.compute_intra_geometrics(ft_preds)
        length_targets, dot_targets, cross_targets = self.compute_intra_geometrics(ft_targets)
        isnotnan = torch.isfinite(length_targets)
        intra_loss += self.loss_geo(length_preds[isnotnan], length_targets[isnotnan], weight, reduction=reduction)
        isnotnan = torch.isfinite(dot_targets)
        intra_loss += self.loss_geo(dot_preds[isnotnan], dot_targets[isnotnan], weight, reduction=reduction)
        isnotnan = torch.isfinite(cross_targets)
        intra_loss += self.loss_geo(cross_preds[isnotnan], cross_targets[isnotnan], weight, reduction=reduction)

        # inter, relation
        inter_loss = 0
        re_preds = pred.view(-1, self.num_ins, self.num_pts, 2)
        re_targets = normalized_target.view(-1, self.num_ins, self.num_pts, 2)
        re_labels = labels.view(-1 ,self.num_ins)

        offset_re_preds = re_preds - torch.roll(re_preds, shifts=1, dims=2)
        offset_re_targets = re_targets - torch.roll(re_targets, shifts=1, dims=2)
        for idx in range(re_preds.shape[0]):
            ft_preds = re_preds[idx][re_labels[idx] < self.num_classes]
            ft_targets = re_targets[idx][re_labels[idx] < self.num_classes]
            ft_offset_preds = offset_re_preds[idx][re_labels[idx] < self.num_classes]
            ft_offset_targets = offset_re_targets[idx][re_labels[idx] < self.num_classes]
            length_preds, dot_preds, cross_preds = self.compute_inter_geometrics(ft_preds, ft_offset_preds)
            length_targets, dot_targets, cross_targets = self.compute_inter_geometrics(ft_targets, ft_offset_targets)
            isnotnan = torch.isfinite(length_targets)
            inter_loss += self.loss_geo(length_preds[isnotnan], length_targets[isnotnan], weight, reduction=reduction)
            isnotnan = torch.isfinite(dot_targets)
            inter_loss += self.loss_geo(dot_preds[isnotnan], dot_targets[isnotnan], weight, reduction=reduction)
            isnotnan = torch.isfinite(cross_targets)
            inter_loss += self.loss_geo(cross_preds[isnotnan], cross_targets[isnotnan], weight, reduction=reduction)
        if self.inter_loss_weight is not None or self.intra_loss_weight is not None:
            loss = 0
            if self.inter_loss_weight is not None:
                loss += self.inter_loss_weight * inter_loss
            else:
                loss += inter_loss
            if self.intra_loss_weight is not None:
                loss += self.intra_loss_weight * intra_loss
            else:
                loss += intra_loss
        else:
            loss = intra_loss + inter_loss
        return loss * self.loss_weight