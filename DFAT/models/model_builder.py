# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

from DFAT.core.config import cfg
from DFAT.models.backbone import get_backbone
from DFAT.models.head import get_rpn_head, get_mask_head, get_refine_head
from DFAT.models.neck import get_neck
from DFAT.models.RFN import mRFN
from DFAT.models.loss import  weight_l1_loss
from DFAT.models.loss import select_cross_entropy_loss, select_iou_loss, feature_loss, spa_loss, ssim_loss

EPSILON = 1e-10
RGB_RATE = 0.8
IR_RATE = 0.8
FLAG = 1
SELECTION = 'max'
SPATIAL = 'max'
POOLING = 'mean'
scale = 0.5
W_rgb = 0.5
W_ir = 0.5
import time
class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)


        if cfg.RFN.RFN:
            # self.RFN_s = mRFN(cfg.RFN.Input, 3)
            # self.RFN_t = mRFN(cfg.RFN.Input, 3)
            self.RFN = mRFN(cfg.RFN.Input, 3)

    def get_from_data(self, data):
        template = [data['template'][i].cuda() for i in range(len(data['template']))]
        label_mask = data['label_mask'].cuda()
        label_mask_weight = data['label_mask_weight'].cuda()

        if cfg.ANCHOR.TYPE:
            label_loc_weight = [data['label_loc_weight'][i].cuda() for i in range(len(data['label_loc_weight']))]
        else:
            label_loc_weight = [data['label_loc_weight'][i].cuda() for i in range(len(data['label_loc_weight']))]

        search = [data['search'][i].cuda() for i in range(len(data['search']))]
        label_cls = [data['label_cls'][i].cuda() for i in range(len(data['label_cls']))]
        label_loc = [data['label_loc'][i].cuda() for i in range(len(data['label_loc']))]
        sb = data['bbox']
        # tb = data['template_bbox']
        return template, search, label_cls, label_loc, label_loc_weight, label_mask, label_mask_weight, sb

    def forward(self, data):
        """ only used in training
        """

        # index = int((cfg.TIME.SEARCH_NUM - 1) / 2)
        template, search, label_cls, label_loc, label_loc_weight, label_mask, label_mask_weight, sea_bbox = self.get_from_data(data)

        zf_rgb = self.backbone(template[0])  # sames sparse
        zf_tir = self.backbone(template[1])  # sames sparse
        # zf_tir = self.backbone_rgb(template[1])  # sames sparse
        xf_rgb = self.backbone(search[0])  # sames sparse
        xf_tir = self.backbone(search[1])
        # xf_tir = self.backbone_rgb(search[1])
        if cfg.ADJUST.ADJUST:
            zf_rgb = self.neck(zf_rgb)
            xf_rgb = self.neck(xf_rgb)
        if cfg.ADJUST.ADJUST:
            zf_tir = self.neck(zf_tir)
            xf_tir = self.neck(xf_tir)
            # zf_tir = self.neck_rgb(zf_tir)
            # xf_tir = self.neck_rgb(xf_tir)
        if not cfg.RFN.RFN:
            cls_rgb, loc_rgb = self.rpn_head(zf_rgb, xf_rgb)
            cls_tir, loc_tir = self.rpn_head(zf_tir, xf_tir)

            cls_loss_rgb = self.get_cls_loss(cls_rgb, label_cls[0])
            cls_loss_tir = self.get_cls_loss(cls_tir, label_cls[1])
            cls_loss = cls_loss_rgb * cfg.TRAIN.cls_weight + cls_loss_tir * (1 - cfg.TRAIN.cls_weight)

            loc_loss_rgb = self.get_loc_loss(loc_rgb, label_loc[0], label_loc_weight[0], label_cls[0])
            loc_loss_tir = self.get_loc_loss(loc_tir, label_loc[1], label_loc_weight[1], label_cls[1])#, loc_threshold=cfg.TRAIN.loc_threshold
            loc_loss = loc_loss_rgb * cfg.TRAIN.loc_weight + loc_loss_tir * (1 - cfg.TRAIN.loc_weight)
            outputs = {}
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss
            total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                         cfg.TRAIN.LOC_WEIGHT * loc_loss
            outputs['total_loss'] = total_loss

            return outputs
        else:
            zf_spa = self.RFN.spatial_model(zf_rgb, zf_tir)
            zf = self.RFN(zf_rgb, zf_tir)
            xf = self.RFN(xf_rgb, xf_tir)
            xf_spa = self.RFN.spatial_model(xf_rgb, xf_tir)
            cls, loc = self.rpn_head(zf, xf)

            # cls_loss_rgb = self.get_cls_loss(cls, label_cls[0])
            cls_loss_tir = self.get_cls_loss(cls, label_cls[0])
            # cls_loss = cls_loss_rgb * cfg.TRAIN.cls_weight + cls_loss_tir * (1 - cfg.TRAIN.cls_weight)
            cls_loss = cls_loss_tir

            # loc_loss_rgb = self.get_loc_loss(loc_rgb, label_loc[0], label_loc_weight[0], label_cls[0])
            loc_loss_tir = self.get_loc_loss(loc, label_loc[0], label_loc_weight[0],
                                             label_cls[0])  # , loc_threshold=cfg.TRAIN.loc_threshold
            loc_loss = loc_loss_tir

            outputs = {}
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss

            fea_loss_z1 = feature_loss(zf_rgb, zf_tir, zf)
            fea_loss_z2 = spa_loss(zf_spa, zf)
            fea_loss_z3 = ssim_loss(zf_rgb, zf)
            fea_loss_x1 = feature_loss(xf_rgb, xf_tir, xf)
            fea_loss_x2 = spa_loss(xf_spa, xf)
            fea_loss_x3 = ssim_loss(xf_rgb, xf)

            rfn_two_loss = fea_loss_z1 + fea_loss_x1
            rfn_spa_loss = fea_loss_z2 + fea_loss_x2
            rfn_ssim_loss = fea_loss_z3 + fea_loss_x3
            fea_loss = 0 * rfn_spa_loss + rfn_two_loss + rfn_ssim_loss

            outputs['rfn_two'] = rfn_two_loss
            outputs['rfn_spa'] = 0 * rfn_spa_loss
            outputs['rfn_ssim'] = rfn_ssim_loss
            outputs['fea_loss'] = fea_loss

            outputs['total_loss'] = 1.0 * cls_loss + \
                                    1.2 * loc_loss + \
                                    0.1 * fea_loss
            # outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            #                         cfg.TRAIN.LOC_WEIGHT * loc_loss + \
            #                         cfg.TRAIN.MASK_WEIGHT * mask_loss

            return outputs
        # outputs = {}
        # outputs['cls_loss'] = cls_loss
        # outputs['loc_loss'] = loc_loss
        # total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
        #              cfg.TRAIN.LOC_WEIGHT * loc_loss
        # outputs['total_loss'] = total_loss
        #
        # return outputs

    def get_cls_loss(self, cls, label_cls):
        cls = self.log_softmax(cls)  # 48x5x25x25x2
        cls_loss, cls_loss_pos, cls_loss_neg = select_cross_entropy_loss(cls, label_cls)
        return cls_loss

    def get_loc_loss(self, loc, label_loc, label_loc_weight, label_cls, loc_threshold=0.0):
        if cfg.ANCHOR.TYPE:
            loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
            # loc_loss = select_iou_loss(loc, label_loc, label_cls)
        else:
            # loc_loss = select_iou_loss(loc, label_loc, label_cls, loc_threshold=loc_threshold)
            loc_loss = 0.

        return loc_loss


    #attention
    #opt_attention
    def opt_attention(self, ori, att, att_type='S'):
        num = len(ori)
        out = []
        if att_type == 'S':
            for i in range(num):
                out.append(self.opt_spatial_attention(ori[i], att[i]))
        else:
            for i in range(num):
                out.append(self.opt_channel_attention(ori[i], att[i]))
        return out

    #opt_spatial_attention
    def opt_spatial_attention(self, ori, att):
        sh = ori.shape
        out = ori.contiguous().view(-1, sh[1]).__mul__(
            att).view(ori.shape)
        return out

    def opt_channel_attention(self, ori, att):
        sh = ori.shape
        out = ori.contiguous().view(sh[1], -1).__mul__(
            att).view(ori.shape)
        return out

    #get_spatial attention
    def get_spatial_attention(self, fea, spatial_type='mean'):
        num = len(fea)
        att = []
        for i in range(num):
            att.append(self.spatial_attention(fea[i], spatial_type=spatial_type))
        return att

    def get_channel_attention(self, fea, spatial_type='mean'):
        num = len(fea)
        att = []
        for i in range(num):
            att.append(self.channel_attention(fea[i], spatial_type=spatial_type))
        return att

    def spatial_attention(self, tensor, spatial_type='mean', act=True):
        spatial = None
        if spatial_type == 'mean':
            spatial = tensor.mean(dim=1, keepdim=True)
        elif spatial_type == 'sum':
            spatial = tensor.sum(dim=1, keepdim=True)
        elif spatial_type == 'max':
            spatial, _ = tensor.max(dim=1, keepdim=True)
        return spatial.contiguous().view(-1, 1)
        #return F.sigmoid(spatial.contiguous().view(-1, 1))
        #return self.batch_nor(spatial.contiguous().view(-1, 1))

    def channel_attention(self, tensor, spatial_type='mean', act=True):
        chan = None
        sh = tensor.shape
        if spatial_type == 'mean':
            chan = tensor.contiguous().view(sh[1], -1).mean(dim=1, keepdim=True)
        elif spatial_type == 'sum':
            chan = tensor.contiguous().view(sh[1], -1).sum(dim=1, keepdim=True)
        elif spatial_type == 'max':
            chan, _ = tensor.contiguous().view(sh[1], -1).max(dim=1, keepdim=True)

        return chan
        # return F.sigmoid(chan)
    # for final fusion in decision level
    def _get_w_fusion_spatial(self, tensor_rgb, tensor_ir, spatial_type='mean', act_f=True):
        # type = 'mean'
        shape = tensor_rgb[0].size()
        num = len(tensor_rgb)
        w_rgb = 0.
        w_ir = 0.
        for i in range(num):
            # spatial attention
            spatial1 = self.spatial_attention(tensor_rgb[i], spatial_type=spatial_type, act=act_f)
            spatial2 = self.spatial_attention(tensor_ir[i], spatial_type=spatial_type, act=act_f)
            # get weight map, soft-max
            # v1 = torch.max(spatial1)
            # v2 = torch.max(spatial2)
            v1 = torch.mean(torch.abs(spatial1))
            v2 = torch.mean(torch.abs(spatial2))
            # torch.exp
            w_1 = (v1) / ((v1) + (v2) + EPSILON)
            w_2 = (v2) / ((v1) + (v2) + EPSILON)

            w_rgb = w_rgb + w_1
            w_ir = w_ir + w_2
        w_rgb = w_rgb / num
        w_ir = w_ir / num

        return [w_rgb, w_ir]
    #get fused feature
    #pooling_type for feature selection, att_type to decide spatial/channel attention, spatial_type to decide to use max/average pooling for att_type, selection_type to get the weights from the attention map
    def fea_fusion(self, rgb, ir, rgb_rate, ir_rate, att_type='S', selection_type='mean', pooling_type='mean', spatial_type='mean', flag=0):
        num = len(rgb)
        out = []
        w_r = 0.
        w_i = 0.
        all = 0.
        feature_rgb = []
        feature_ir = []
        #channel selection
        if flag == 1:
            for j in range(num):
                a, b = self.channel_selection(rgb[j], ir[j], rgb_rate, ir_rate,
                                                                pooling_type=pooling_type)
                feature_rgb.append(a)
                feature_ir.append(b)
        else:
            feature_rgb = rgb
            feature_ir = ir

        #get attention
        if att_type == 'S':
            r = self.get_spatial_attention(feature_rgb, spatial_type)
            i = self.get_spatial_attention(feature_ir, spatial_type)
        else:
            r = self.get_channel_attention(feature_rgb, spatial_type)
            i = self.get_channel_attention(feature_ir, spatial_type)

        for j in range(num):
            if selection_type == 'mean':
                w_r = torch.mean(r[j])
                w_i = torch.mean(i[j])
            else:
                w_r = torch.max(r[j])
                w_i = torch.max(i[j])
            all = w_r + w_i
            out.append(((w_r / all) * rgb[j] + (w_i / all) * ir[j]))
        return out
    # channel selection
    def channel_selection(self, tensor_rgb, tensor_ir, rate_rgb, rate_ir, pooling_type='mean'):
        shape = tensor_rgb.size()
        zero = torch.zeros(1).cuda()
        one = torch.ones(1).cuda()
        # average global pooling
        if pooling_type == 'mean':
            POOL = nn.AdaptiveAvgPool2d((1))
        elif pooling_type == 'max':
            POOL = nn.AdaptiveMaxPool2d((1))
        # RGB tensor selection
        # precentage, 95%
        index_range = int(shape[1] * rate_rgb)
        if index_range is not shape[1]:
            channel_one_dim = POOL(tensor_rgb)
            channel_one_dim = torch.squeeze(channel_one_dim)
            # find max
            mask_dim_rgb = []
            for batch_index in range(shape[0]):
                if shape[0] > 1:
                    channel_one_temp = channel_one_dim[batch_index, :]
                else:
                    channel_one_temp = channel_one_dim
                _, index = torch.sort(channel_one_temp, descending=True)
                index = torch.squeeze(index)
                index_min = index[index_range - 1]
                th_min = channel_one_temp[index_min]
                mask_temp = torch.where(channel_one_temp >= th_min, one, zero)
                mask_temp = mask_temp.reshape(1, mask_temp.size()[0], 1, 1)
                mask_temp = mask_temp.repeat(1, 1, shape[2], shape[3])
                if batch_index is 0:
                    mask_dim_rgb = mask_temp
                else:
                    mask_dim_rgb = torch.cat([mask_dim_rgb, mask_temp], 0)
            tensor_rgb_select = mask_dim_rgb * tensor_rgb
        else:
            tensor_rgb_select = tensor_rgb

        # IR tensor selection
        # precentage, 75%
        index_range_ir = int(shape[1] * rate_ir)
        if index_range_ir is not shape[1]:
            channel_one_dim_ir = POOL(tensor_ir)
            channel_one_dim_ir = torch.squeeze(channel_one_dim_ir)
            mask_dim_ir = []
            for batch_index in range(shape[0]):
                if shape[0] > 1:
                    channel_one_temp = channel_one_dim_ir[batch_index, :]
                else:
                    channel_one_temp = channel_one_dim_ir
                _, index_ir = torch.sort(channel_one_temp, descending=True)
                index_ir = torch.squeeze(index_ir)
                # find min th
                index_min_ir = index_ir[index_range_ir - 1]
                th_min_ir = channel_one_temp[index_min_ir]
                # get mask
                mask_ir_temp = torch.where(channel_one_temp >= th_min_ir, one, zero)
                mask_ir_temp = mask_ir_temp.reshape(1, mask_ir_temp.size()[0], 1, 1)
                mask_ir_temp = mask_ir_temp.repeat(1, 1, shape[2], shape[3])
                if batch_index is 0:
                    mask_dim_ir = mask_ir_temp
                else:
                    mask_dim_ir = torch.cat([mask_dim_ir, mask_ir_temp], 0)
            # set 0
            tensor_ir_select = mask_dim_ir * tensor_ir
        else:
            tensor_ir_select = tensor_ir

        return tensor_rgb_select, tensor_ir_select
    # template update
    def templateupdate(self, z):
        #z_rgb = z_update[0]
        #z_ir = z_update[1]
        # rgb
        zf_rgb = self.backbone(z[0])
        zf_ir = self.backbone(z[1])
        if cfg.REFINE.REFINE:
            zf_rgb = zf_rgb[2:]
            zf_ir = zf_ir[2:]
        if cfg.MASK.MASK and cfg.RPN.TYPE not in ['MultiRPN']:
            zf_rgb = zf_rgb[-1]
            zf_ir = zf_ir[-1]
        if cfg.ADJUST.ADJUST:
            zf_rgb = self.neck(zf_rgb)
            zf_ir = self.neck(zf_ir)

        # update use fused features
        #self.zf_rgb = [(zf_rgb[i] * 0.1 + self.zf_rgb[i] * 0.9) for i in range(3)]
        #self.zf_ir = [(zf_ir[i] * 0.1 + self.zf_ir[i] * 0.9) for i in range(3)]
        self.zf_rgb = [torch.from_numpy(zf_rgb[i].detach().cpu().numpy().reshape([256,-1]) * 0.1 + self.zf_rgb[i].detach().cpu().numpy().reshape([256,-1]) * 0.9).view(zf_rgb[i].shape).cuda() for i in range(3)]
        self.zf_ir = [torch.from_numpy(zf_ir[i].detach().cpu().numpy().reshape([256,-1]) * 0.1 + self.zf_ir[i].detach().cpu().numpy().reshape([256,-1]) * 0.9).view(zf_ir[i].shape).cuda() for i in range(3)]


        #self.zf_fused = self.fea_fusion(self.zf_rgb, self.zf_ir, RGB_RATE, IR_RATE, 'S', selection_type=SELECTION, pooling_type=POOLING, spatial_type=SPATIAL, flag=FLAG)
        # self.zf_fused = self.fea_fusion(self.zf_rgb, self.zf_ir, RGB_RATE, IR_RATE, 'C', selection_type=SELECTION, pooling_type=POOLING, spatial_type=SPATIAL, flag=FLAG)


    def template(self, img):

        z_rgb = self.backbone(img[0])
        z_ir = self.backbone(img[1])

        if cfg.REFINE.REFINE:
            z_rgb = z_rgb[2:]
            z_ir = z_ir[2:]
        if cfg.MASK.MASK and cfg.RPN.TYPE not in ['MultiRPN']:
            z_rgb = z_rgb[-1]
            z_ir = z_ir[-1]
        if cfg.ADJUST.ADJUST:
            z_rgb = self.neck(z_rgb)
            z_ir = self.neck(z_ir)

        # for update
        self.zf_rgb = z_rgb
        self.zf_ir = z_ir
        self.zf_rgb_ori = z_rgb
        self.zf_ir_ori = z_ir
        self.gt = None
        self.bbox = None
        self.pre_s_rgb = None
        self.pre_s_ir = None
        self.count = 0
        self.pre_x_rgb = z_rgb
        self.pre_x_ir = z_ir
        self.rgb_txt = None
        self.ir_txt = None


        #self.zf_fused = self.fea_fusion(self.zf_rgb, self.zf_ir, RGB_RATE, IR_RATE, 'S', selection_type=SELECTION, pooling_type=POOLING, spatial_type=SPATIAL, flag=FLAG)
        # self.zf_fused = self.fea_fusion(self.zf_rgb, self.zf_ir, RGB_RATE, IR_RATE, 'C', selection_type=SELECTION, pooling_type=POOLING, spatial_type=SPATIAL, flag=FLAG)
        # self.mmm1 = []
        # self.mmm1 = np.zeros([226, 1])
        # self.mmm2 = np.zeros([226, 1])

        if cfg.RFN.RFN:
            self.zf = self.RFN(self.zf_rgb, self.zf_ir)

    def data_mean(self, rgb, ir):
        r = rgb > 0
        i = ir > 0
        m_r = torch.sum(rgb * r.float()) / (len(torch.nonzero(rgb * r.float())) + EPSILON)
        m_i = torch.sum(ir * i.float()) / (len(torch.nonzero(ir * i.float())) + EPSILON)
        # return [m_i, m_r]
        all = (m_r + m_i)
        dis = m_r - m_i
        # if (dis != 0) and (m_r !=0) and (m_i != 0):
        #     return [all / m_r, all / m_i]
        # elif (m_r == 0) and (m_i !=0) and (dis != 0):
        #     return [0., 1.]
        # elif (m_i == 0) and (m_r != 0) and (dis != 0):
        #     return [0, 1]
        # else:
        #     return [0.5, 0.5]
        # print(m_r)
        # print(m_i)
        if dis == 0:
            return [0.5, 0.5]
        elif m_r == 0 and m_i ==0:
            return [0.5, 0.5]
        elif m_r == 0:
            return [0, 1]
        elif m_i == 0:
            return [1, 0]
        else:
            return [0.5, 0.5]# [all / m_r, all / m_i]

    def track(self, img):
        #num = len(x)
        # num = 1
        self.count = self.count + 1
        # tic = time.time()
        x_rgb = self.backbone(img[0])
        x_ir = self.backbone(img[1])
        # print('Backbone casts %f' % (time.time() - tic))

        if cfg.REFINE.REFINE:
            self.x_rgb = x_rgb[:-2]
            x_rgb = x_rgb[2:]
            x_ir = x_ir[2:]
        if cfg.MASK.MASK and cfg.RPN.TYPE not in ['MultiRPN']:
            self.x_rgb = x_rgb[:-1]
            x_rgb = x_rgb[-1]
            x_ir = x_ir[-1]
        if cfg.ADJUST.ADJUST:
            # tic = time.time()
            x_rgb = self.neck(x_rgb)
            x_ir = self.neck(x_ir)
            # print('Neck casts %f' % (time.time() - tic))

        #feature-level
        #spatial
        # x_s_r = self.get_spatial_attention(x_rgb, 'mean')
        # z_s_r = self.get_spatial_attention(self.zf_rgb, 'mean')
        # x_s_i = self.get_spatial_attention(x_ir, 'mean')
        # z_s_i = self.get_spatial_attention(self.zf_ir, 'mean')
        #
        # z_s_r_done = self.opt_attention(self.zf_rgb, z_s_r, 'S')
        # z_s_i_done = self.opt_attention(self.zf_ir, z_s_i, 'S')
        # x_s_r_done = self.opt_attention(x_rgb, x_s_r, 'S')
        # x_s_i_done = self.opt_attention(x_ir, x_s_i, 'S')
        #
        # cls_rgb, loc_rgb = self.rpn_head(self.zf_rgb_done, x_s_r_done)
        # cls_ir, loc_ir = self.rpn_head(self.zf_ir_done, x_s_i_done)

        #channel
        # x_c_r = self.get_channel_attention(x_rgb, 'mean')
        # z_c_r = self.get_channel_attention(self.zf_rgb, 'mean')
        # x_c_i = self.get_channel_attention(x_ir, 'mean')
        # z_c_i = self.get_channel_attention(self.zf_ir, 'mean')
        #
        # z_c_r_done = self.opt_attention(self.zf_rgb, z_c_r, 'C')
        # z_c_i_done = self.opt_attention(self.zf_ir, z_c_i, 'C')
        # x_c_r_done = self.opt_attention(x_rgb, x_c_r, 'C')
        # x_c_i_done = self.opt_attention(x_ir, x_c_i, 'C')
        #
        # cls_rgb, loc_rgb = self.rpn_head(z_c_r_done, x_c_r_done)
        # cls_ir, loc_ir = self.rpn_head(z_c_i_done, x_c_i_done)


        #fusion-level_fused
        #selection_type --> choose value in attention matrix
        #pooling_type ---> channel selection  (maxpooling/averagepooling)
        #spatial_type ----> obtain attention
        #xf = self.fea_fusion(x_rgb, x_ir, RGB_RATE, IR_RATE, 'S', selection_type=SELECTION, pooling_type=POOLING, spatial_type=SPATIAL, flag=FLAG)
        # xf = self.fea_fusion(x_rgb, x_ir, RGB_RATE, IR_RATE, 'C', selection_type=SELECTION, pooling_type=POOLING, spatial_type=SPATIAL, flag=FLAG)
        #cls, loc = self.rpn_head(self.zf_fused, xf)

        #if cfg.MASK.MASK:
        #    mask = self.mask_head(self.zf_rgb, x_rgb)
        #if cfg.REFINE.REFINE:
        #    self.mask_corr_feature = self.mask_head.corrfeature(self.zf_rgb, x_rgb)
        #return {
        #        'cls': cls,
        #        'loc': loc,
        #        'mask': mask if cfg.MASK.MASK else None
        #       }


        if not cfg.RFN.RFN:
            # tic = time.time()
            cls_rgb, loc_rgb = self.rpn_head(self.zf_rgb, x_rgb)
            cls_ir, loc_ir = self.rpn_head(self.zf_ir, x_ir)
            # print('RPN casts %f' % (time.time() - tic))
            # decision-level
            #
            # print(torch.max(cls_rgb.view(-1, 1)))
            # print(torch.max(cls_ir.view(-1, 1)))
            # print(self.count)
            # print('\n')
            # with open(self.rgb_txt, 'a') as f1:
            #     f1.writelines(str(torch.max(cls_rgb)))
            #     f1.writelines('\n')
            #     f1.close()
            # with open(self.ir_txt, 'a') as f2:
            #     f2.writelines(str(torch.max(cls_ir)))
            #     f2.writelines('\n')
            #     f2.close()

            # self.mmm1.append(torch.max(cls_rgb.view(-1, 1)))
            # self.mmm1[self.count] = torch.max(cls_rgb.view(-1, 1))
            # self.mmm2[self.count] = torch.max(cls_ir.view(-1, 1))
            # fusion
            # psr
            # w_ir = (np.max(self._convert_score(cls_ir))-np.mean(self._convert_score(cls_ir)))/np.std(self._convert_score(cls_ir))
            # w_rgb = (np.max(self._convert_score(cls_rgb)) - np.mean(self._convert_score(cls_rgb))) / np.std(
            #     self._convert_score(cls_rgb))

            # updt
            # w_ir = self.udpt(self._convert_score(cls_rgb), self._convert_score(cls_ir), 0.3, 0.8)
            # w_rgb = 1 - w_ir

            # channel attention
            # w = self._get_w_fusion_spatial(x_rgb, x_ir)
            # w_rgb = w[0]
            # w_ir = w[1]

            # similarity with model
            # mo = 'all'  # now/ori/n_s/o_s/n_o/all/pre_s
            # w = self.get_simi_model(x_rgb, x_ir, mo)
            # w_rgb = w[0]
            # w_ir = w[1]

            # data_mean
            # rgb = self._convert_score(cls_rgb)
            # ir = self._convert_score(cls_ir)
            # w = self.data_mean(rgb, ir)
            # w_rgb = w[0]
            # w_ir = w[1]

            # data_mean
            # tic = time.time()
            w = self.data_mean(self._convert_score(cls_rgb), self._convert_score(cls_ir))

            # fusion_pixel----here must set 0.5/0.5
            # w_rgb = w[0]
            # w_ir = w[1]

            # i = (w_ir * cls_ir)/((w_rgb + w_ir))
            # r = (w_rgb * cls_rgb)/((w_rgb + w_ir))
            #
            # with open(self.rgb_txt, 'a') as f1:
            #     f1.writelines(str(torch.max(r)))
            #     f1.writelines('\n')
            #     f1.close()
            # with open(self.ir_txt, 'a') as f2:
            #     f2.writelines(str(torch.max(i)))
            #     f2.writelines('\n')
            #     f2.close()
            # print(self.count)
            # print(w)

            cls = scale * (w[1] * cls_ir + w[0] * cls_rgb) / ((w[1] + w[0]))
            # w_rgb = 0.5
            # w_ir = 0.5
            # cls = (w_ir * cls_ir + w_rgb * cls_rgb)

            loc = (0.5 * loc_ir + 0.5 * loc_rgb)
            # toc = time.time()
            # print('DLF casts %f' % (toc- tic))
        else:

            search = self.RFN(x_rgb, x_ir)
            cls, loc = self.rpn_head(self.zf, search)

        #
        if cfg.MASK.MASK:
            mask = self.mask_head(self.zf_rgb, x_rgb)
        if cfg.REFINE.REFINE:
            self.mask_corr_feature = self.mask_head.corrfeature(self.zf_rgb, x_rgb)
        # #
        return {
                'cls': cls,
                'loc': loc,
        #         # 'cls_rgb': cls_rgb,
        #         # 'cls_ir': cls_ir,
        #         # 'loc_rgb': loc_rgb,
        #         # 'loc_ir': loc_ir,
                'mask': mask if cfg.MASK.MASK else None
               }

    #get the positive score
    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        #score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        # score = score.data[:, 1].cpu().numpy()
        score = score.data[:, 1]
        return score

    #emurate -- updt
    def udpt(self, cls_rgb, cls_ir, start=0.4, end=0.6, step=0.01):
        len = int((end - start)/step) + 1
        ma = 0
        tem = torch.zeros(cls_rgb.shape)
        out = 0
        for i in range(len):
            a = i*step+start
            tem = cls_ir * a + cls_rgb * (1-a)
            if torch.max(tem) > ma:
                out = i*step+start
                ma = torch.max(tem)
                # print(out)
                # print("\n")
        # return out
        if out >= 0.5:
            return out
        else:
            return (1-out)

    def get_highscore(self, rgb, ir):
        dis = (rgb - ir) > 0
        r = rgb > 0
        i = ir > 0
        # delta_r= rgb_b.permute(1, 2, 3, 0).contiguous().view(-1, 4)
        # delta_i = ir_b.permute(1, 2, 3, 0).contiguous().view(-1, 4)
        fused_s = rgb * dis.float() * r.float() * i.float() + ir * (1 - dis).float() * r.float() * i.float()
        # d1 = dis.float().unsqueeze(1).repeat(1, 4)
        # r1 = r.float().unsqueeze(1).repeat(1, 4)
        # i1 = i.float().unsqueeze(1).repeat(1, 4)
        # fused_b = delta_r * d1 * r1 * i1 + delta_i * (1 - d1) * r1 * i1
        return fused_s

    def mask_refine(self, pos):
        return self.refine_head(self.x_rgb, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    #get similarity -outter
    def get_simi_model(self, rgb, ir, m='now'):
        w = torch.ones(2).cuda()
        way = 'mean'
        if m == 'now':
            w1 = self.get_simi(rgb, self.zf_rgb, way)
            w2 = self.get_simi(ir, self.zf_ir, way)
            w[0] = w1 / (w1 + w2 + EPSILON)
            w[1] = w2 / (w1 + w2 + EPSILON)
        elif m == 'ori':
            w1 = self.get_simi(rgb, self.zf_rgb_ori, way)
            w2 = self.get_simi(ir, self.zf_ir_ori, way)
            w[0] = w1 / (w1 + w2 + EPSILON)
            w[1] = w2 / (w1 + w2 + EPSILON)
        elif m == 'n_s':
            w1 = self.get_simi(rgb, self.zf_rgb, way)
            w2 = self.get_simi(ir, self.zf_ir, way)
            w3 = self._get_w_fusion_spatial(rgb, ir, way)
            w[0] = w1 / (w1 + w2 + EPSILON)
            w[1] = w2 / (w1 + w2 + EPSILON)
            w = [(w[i] + w3[i])/2 for i in range(2)]
        elif m == 'o_s':
            w1 = self.get_simi(rgb, self.zf_rgb_ori, way)
            w2 = self.get_simi(ir, self.zf_ir_ori, way)
            w3 = self._get_w_fusion_spatial(rgb, ir, way)
            w[0] = w1 / (w1 + w2 + EPSILON)
            w[1] = w2 / (w1 + w2 + EPSILON)
            w = [(w[i] + w3[i]) / 2 for i in range(2)]
        elif m == 'n_o':
            w1 = self.get_simi(rgb, self.zf_rgb, way)
            w2 = self.get_simi(ir, self.zf_ir, way)
            w3 = self.get_simi(rgb, self.zf_rgb_ori, way)
            w4 = self.get_simi(ir, self.zf_ir_ori, way)
            w[0] = (w1 + w3) / (w1 + w2 + w3 + w4 + EPSILON)
            w[1] = (w2 + w4)/ (w1 + w2 + w3 + w4 + EPSILON)
        elif m == 'pre_s':
            w1 = self.get_simi(rgb, self.pre_x_rgb, way)
            w2 = self.get_simi(ir, self.pre_x_ir, way)
            w[0] = w1 / (w1 + w2 + EPSILON)
            w[1] = w2 / (w1 + w2 + EPSILON)
        else:
            w1 = self.get_simi(rgb, self.zf_rgb, way)
            w2 = self.get_simi(ir, self.zf_ir, way)
            w3 = self.get_simi(rgb, self.zf_rgb_ori, way)
            w4 = self.get_simi(ir, self.zf_ir_ori, way)
            w5 = self._get_w_fusion_spatial(rgb, ir, way)
            w[0] = (w1 + w3) / (w1 + w2 + w3 + w4 + EPSILON)
            w[1] = (w2 + w4) / (w1 + w2 + w3 + w4 + EPSILON)
            w = [(w[i] + w5[i]) / 2 for i in range(2)]
        return w
    #get similarity -inner
    def get_simi(self, sample, model, way):
        s1 = sample[0].shape
        s2 = model[0].shape
        num = len(sample)
        dis = []
        v_s = torch.ones(num)
        v_m = torch.ones(num)
        for i in range(num):
            v_s[i] = torch.mean(torch.abs(self.spatial_attention(sample[i], way)))
            v_m[i] = torch.mean(torch.abs(self.spatial_attention(model[i], way)))
        dis = torch.abs(v_s - v_m)
        return torch.mean(dis)

    def get_prebbox(self):
        return self.bbox

    def set_prebbox(self, bbox):
        self.bbox = bbox
        # return 0

    def get_prescore(self):
        return self.pre_s_rgb, self.pre_s_ir, self.count

    def set_prescore(self, s_rgb, s_ir):
        self.pre_s_rgb = s_rgb
        self.pre_s_ir = s_ir
        return 0
    def set_pre_x(self, x_rgb, x_ir):
        self.pre_x_rgb = x_rgb
        self.pre_x_ir = x_ir
        return 0

    def set_fileroot(self, rgb, ir):
        self.rgb_txt = rgb
        self.ir_txt = ir
        # return 0
    def batch_nor(self, input):
        m = torch.mean(input)
        s = torch.std(input)
        #out = (input - m) / (s + EPSILON)
        ma = torch.max(input)
        mi = torch.min(input)
        return (input - mi) / (ma - mi)