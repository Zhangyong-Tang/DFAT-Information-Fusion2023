# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from DFAT.core.config import cfg
from DFAT.models.iou_loss import linear_iou
import DFAT.models.pytorch_msssim as py_msssim
mse_loss = nn.MSELoss()
EPSILON = 1e-5
def feature_loss(rgb_fea, ir_fea, f_fea):
    loss_value = 0.
    # feature loss
    w_ir = [4.0, 4.0, 4.0]
    w_vi = [0.5, 0.5, 0.5]
    w_fea = [1, 1, 1]
    for rgb, ir, f, w1, w2, w3 in zip(rgb_fea, ir_fea, f_fea, w_fea, w_ir, w_vi):
        (bt, cht, ht, wt) = rgb.size()
        loss_value += w1 * mse_loss(f, w2 * ir + w3 * rgb) / (cht * ht * wt)
    return loss_value


def spa_loss(spa_fea, f_fea):
    loss_value = 0.
    # feature loss
    for f, spa in zip(f_fea, spa_fea):
        (bt, cht, ht, wt) = f.size()
        loss_value += mse_loss(f, spa) / (cht * ht * wt)

    return loss_value


def ssim_loss(rgb_fea, f_fea):
    loss_value = 0.
    bt, cht, ht, wt = f_fea[0].size()
    for rgb, f in zip(rgb_fea, f_fea):
        f = f.contiguous().view(1, bt, -1, ht * wt) + EPSILON
        rgb = rgb.contiguous().view(1, bt, -1, ht * wt) + EPSILON
        # value = py_msssim.msssim(f, rgb, normalize=False)
        value = py_msssim.ssim(f, rgb)
        loss_value += (1 - value)
    return loss_value

def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().cuda()

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 4, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)

    return linear_iou(pred_loc, label_loc)

def get_cls_loss(pred, label, select):
    # if len(select.size()) == 0 or \
    #         select.size() == torch.Size([0]):
    #     return 0
    if select.nelement() == 0: return pred.sum() * 0.
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5, loss_pos, loss_neg


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    # if len(pos.size()) == 0 or pos.size() == torch.Size([0]):
    #     return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0
    if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

    if len(p_m.shape) == 4:
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        p_m = F.interpolate(p_m, size=[g_sz, g_sz])
        p_m = p_m.view(-1, g_sz * g_sz)
    else:
        p_m = torch.index_select(p_m, 0, pos)

    if cfg.REFINE.REFINE:
        mask_uf = F.unfold(mask, (g_sz, g_sz), padding=0, stride=8)
    else:
        mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)

    mask_uf = torch.index_select(mask_uf, 0, pos)
    loss = F.soft_margin_loss(p_m, mask_uf)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label):
    pred = pred.ge(0)
    mask_sum = pred.eq(1).add(label.eq(1))
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn / union
    return torch.mean(iou), (torch.sum(iou > 0.5).float() / iou.shape[0]), (torch.sum(iou > 0.7).float() / iou.shape[0])
