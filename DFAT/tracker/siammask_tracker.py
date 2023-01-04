# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from DFAT.core.config import cfg
from DFAT.utils.bbox import cxy_wh_2_rect
from DFAT.tracker.siamrpn_tracker import SiamRPNTracker
from toolkit.utils.region import vot_overlap

EPSILON = 1e-10
class SiamMaskTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamMaskTracker, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTracker must have mask_head"
        #assert hasattr(self.model, 'refine_head'), \
        #    "SiamMaskTracker must have refine_head"

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img


    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x = round(s_x)

        x_crop = []
        x_crop_rgb = self.get_subwindow(img[0],
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average_rgb)
        x_crop_ir = self.get_subwindow(img[1],
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average_ir)

        m_r = torch.mean(x_crop_rgb)
        m_i = torch.mean(x_crop_ir)
        dis = m_r - m_i
        x_crop = [x_crop_rgb, x_crop_ir + dis]
        # x_crop = [x_crop_rgb, x_crop_ir]
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

        #track
        outputs = self.model.track(x_crop)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        #score_rgb = self._convert_score(outputs['cls_rgb'])
        #score_ir = self._convert_score(outputs['cls_ir'])
        #pred_rgb = self._convert_bbox(outputs['loc_rgb'], self.anchors)
        #pred_ir = self._convert_bbox(outputs['loc_ir'], self.anchors)
        #
        # get rgb final score
        #s_c_rgb = change(sz(pred_rgb[2, :], pred_rgb[3, :]) /
        #             (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # aspect ratio penalty
        #r_c_rgb = change((self.size[0] / self.size[1]) /
        #             (pred_rgb[2, :] / pred_rgb[3, :]))
        #penalty_rgb = np.exp(-(r_c_rgb * s_c_rgb - 1) * cfg.TRACK.PENALTY_K)
        #pscore_rgb = penalty_rgb * score_rgb
        # window penalty
        #pscore_rgb = pscore_rgb * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
        #    self.window * cfg.TRACK.WINDOW_INFLUENCE
        #
        # get ir final score
        #s_c_ir = change(sz(pred_ir[2, :], pred_ir[3, :]) /
        #             (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # aspect ratio penalty
        #r_c_ir = change((self.size[0] / self.size[1]) /
        #             (pred_ir[2, :] / pred_ir[3, :]))
        #penalty_ir = np.exp(-(r_c_ir * s_c_ir - 1) * cfg.TRACK.PENALTY_K)
        #pscore_ir = penalty_ir * score_ir
        # window penalty
        #pscore_ir = pscore_ir * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
        #    self.window * cfg.TRACK.WINDOW_INFLUENCE

        # updt
        # w_ir = self.updt(pscore_ir, pscore_rgb,  0.3, 0.8)

        #kl-d
        #w_ir = self.kld(pscore_ir, pscore_rgb, 1)
        #o = self.model.set_prescore(pscore_rgb, pscore_ir)


        # js-d-after penalty
        # w_ir = self.jsd(pscore_ir, pscore_rgb)
        # # pred_bbox = w_ir * pred_ir + (1 - w_ir) * pred_rgb
        # # pscore = w_ir * score_ir + (1 - w_ir) * score_rgb
        # o = self.model.set_prescore(score_rgb, score_ir)
        #
        #
        #w_ir = 0.5
        #a = 0.5
        #a = w_ir
        #pred_bbox = a * pred_ir + (1 - a) * pred_rgb
        #pscore = w_ir * pscore_ir + (1 - w_ir) * pscore_rgb
        #best_idx = np.argmax(pscore)
        #best_score = pscore[best_idx]
        #bbox = pred_bbox[:, best_idx] / scale_z
        #lr = (w_ir * pscore_ir[best_idx] + (1 - w_ir) * penalty_rgb[best_idx]) * best_score * cfg.TRACK.LR



        #product rule
        # idx, flag = self.product_rule(pscore_rgb, pscore_ir)
        # if flag == 2:
        #     best_idx = idx
        #     pscore = pscore_ir
        #     pred_bbox = pred_ir
        #     best_score = score_ir[idx]   #best_score = pscore[idx]
        #     bbox = pred_bbox[:, best_idx] / scale_z
        #     lr = penalty_ir[best_idx] * best_score * cfg.TRACK.LR
        # else:
        #     best_idx = idx
        #     pscore = pscore_rgb
        #     pred_bbox = pred_rgb
        #     best_score = score_rgb[idx]    #best_score = pscore[idx]
        #     bbox = pred_bbox[:, best_idx] / scale_z
        #     lr = penalty_rgb[best_idx] * best_score * cfg.TRACK.LR



        # #mutil-expert
        # #bb of rgb and ir
        # bb_rgb = pred_rgb[:, np.argmax(pscore_rgb)]/scale_z
        # bb_ir = pred_ir[:, np.argmax(pscore_rgb)] / scale_z
        # #bb of updt
        # w_ir_u = self.updt(pscore_ir, pscore_rgb, 0.3, 0.8)
        # pscore_f_u = w_ir_u * pscore_ir + (1 - w_ir_u) * pscore_rgb
        # bb_f_u = (w_ir_u * pred_ir + (1 - w_ir_u) * pred_rgb)[:, np.argmax(pscore_f_u)]
        # # bb of kld
        # w_ir_k = self.kld(pscore_ir, pscore_rgb)
        # pscore_f_k = w_ir_k * pscore_ir + (1 - w_ir_k) * pscore_rgb
        # bb_f_k = (w_ir_k * pred_ir + (1 - w_ir_k) * pred_rgb)[:, np.argmax(pscore_f_k)]
        # o = self.model.set_prescore(pscore_rgb, pscore_ir)
        #
        # bbox, idx = self.mutil_expert([bb_rgb, bb_ir, bb_f_u, bb_f_k])
        # self.model.set_prebbox(bbox)
        # if idx == 0:
        #     best_idx = np.argmax(pscore_rgb)
        #     best_score = pscore_rgb[best_idx]   #best_score = score_rgb[idx]
        #     lr = penalty_rgb[best_idx] * best_score * cfg.TRACK.LR
        # elif idx == 1:
        #     best_idx = np.argmax(pscore_ir)
        #     best_score = pscore_ir[best_idx]    #best_score = score_ir[idx]
        #     lr = penalty_ir[best_idx] * best_score * cfg.TRACK.LR
        # elif idx == 2:
        #     best_idx = np.argmax(pscore_f_u)
        #     best_score = pscore_f_u[best_idx]
        #     lr = (w_ir_u * pscore_ir[best_idx] + (1 - w_ir_u) * penalty_rgb[best_idx]) * best_score * cfg.TRACK.LR
        # else:
        #     best_idx = np.argmax(pscore_f_k)
        #     best_score = pscore_f_k[best_idx]
        #     lr = (w_ir_k * pscore_ir[best_idx] + (1 - w_ir_k) * penalty_rgb[best_idx]) * best_score * cfg.TRACK.LR




        #kl-d before penalty
        # w_ir = self.kld(score_ir, score_rgb, 0)
        # o = self.model.set_prescore(score_rgb, score_ir)
        # pred_bbox = w_ir * pred_ir + (1 - w_ir) * pred_rgb
        # score = w_ir * score_ir + (1 - w_ir) * score_rgb

        # js-d-before penalty
        # w_ir = self.jsd(score_ir, score_rgb)
        # pred_bbox = w_ir * pred_ir + (1 - w_ir) * pred_rgb
        # score = w_ir * score_ir + (1 - w_ir) * score_rgb
        # o = self.model.set_prescore(score_rgb, score_ir)


        score = self._convert_score(outputs['cls'])
        if cfg.ANCHOR.TYPE:
            pred_bbox = self._convert_bbox_anchor(outputs['loc'], self.anchors)
        else:
            pred_bbox = self._convert_bbox_point(outputs['loc'], self.points)

        #
        # # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        #
        # # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        #
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        best_score = score[best_idx]
        bbox = pred_bbox[:, best_idx] / scale_z




        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy,
                                                width, height, img[0].shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        # self.model.set_prebbox(bbox)

        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        if cfg.REFINE.REFINE:
            mask = self.model.mask_refine((delta_y, delta_x)).sigmoid().squeeze()
            out_size = cfg.TRACK.MASK_OUTPUT_SIZE
            mask = mask.view(out_size, out_size).cpu().data.numpy()
        elif cfg.MASK.MASK:
            mask = outputs['mask'][0, :, delta_y, delta_x].sigmoid().squeeze()
            out_size = cfg.TRACK.MASK_OUTPUT_SIZE
            mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img[0].shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        mask_in_img[mask_in_img < cfg.TRACK.MASK_THERSHOLD] = 0
        mask_in_img[mask_in_img >= cfg.TRACK.MASK_THERSHOLD] = 1
        mask_in_img= mask_in_img.astype(np.uint8)
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()
        # o = self.model.setbbox(bbox)
        return {
                'bbox': bbox,
                'best_score': best_score,
                'mask': mask_in_img,
                'polygon': polygon,
               }

    # def update(self, rgb, ir, bbox):
    #     # calculate z crop size
    #     w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
    #     h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
    #     s_z = round(np.sqrt(w_z * h_z))
    #
    #     self.channel_average_rgb = np.mean(rgb, axis=(0, 1))
    #     self.channel_average_ir = np.mean(ir, axis=(0, 1))
    #
    #     # get crop
    #     z_crop_rgb = self.get_subwindow(rgb, self.center_pos,
    #                                     cfg.TRACK.EXEMPLAR_SIZE,
    #                                     s_z, self.channel_average_rgb)
    #     z_crop_ir = self.get_subwindow(ir, self.center_pos,
    #                                    cfg.TRACK.EXEMPLAR_SIZE,
    #                                    s_z, self.channel_average_ir)
    #
    #     self.model.templateupdate(z_crop_rgb, z_crop_ir)
#emurate -- updt
    def updt(self, ps_ir, ps_rgb, start=0.4, end=0.6, step=0.01):
        out = 0
        cur = 0
        m = 0
        tem = np.zeros(ps_rgb.shape)
        leng = (end - start) / step
        for i in range(int(leng)):
            cur = i * step + start
            tem = cur * ps_ir + (1 - cur) * ps_rgb
            if np.max(tem) > m:
                m = np.max(tem)
                out = cur
        return out

    def kld(self, ps_ir, ps_rgb, flag=0):
        pre_rgb, pre_ir, f = self.model.get_prescore()
        if f == 1:
            return 0.5
        else:
            if flag == 0:
                w1 = np.exp(- np.sum(ps_rgb * np.log(ps_rgb / pre_rgb)))
                w2 = np.exp(- np.sum(ps_ir * np.log(ps_ir / pre_ir)))
            else:
                w1 = np.exp(- np.sum(pre_rgb * np.log(pre_rgb / ps_rgb)))
                w2 = np.exp(- np.sum(pre_ir * np.log(pre_ir / ps_ir)))
            return w2/(w1 + w2 + EPSILON)

    def jsd(self, ps_ir, ps_rgb):
        pre_rgb, pre_ir, f = self.model.get_prescore()
        if f == 1:
            return 0.5
        else:
            w1 = np.exp(- (np.sum(ps_rgb * np.log((ps_rgb + pre_rgb) / 2)) + np.sum(pre_rgb * np.log((ps_rgb + pre_rgb) / 2))) / 2)
            w2 = np.exp(- (np.sum(ps_ir * np.log((ps_ir + pre_ir) / 2)) + np.sum(pre_ir * np.log((ps_ir + pre_ir) / 2))) / 2)
            return w2/(w1 + w2 + EPSILON)

    def mutil_expert(self, bb):
        pre_bb = self.model.get_prebbox()
        m = 0
        idx = 0
        for i in range(len(bb)):
            iou = self.get_iou(pre_bb, bb[i])
            if iou > m:
                idx = i
                m = iou
        return bb[idx], idx

    #get iou
    def get_iou(self, pre, cur):
        p = [(self.center_pos[0] + pre[0] + 0.5 - pre[2] / 2), (self.center_pos[1] + pre[1] + 0.5 - pre[3] / 2), pre[2], pre[3]]
        #no overlap
        if (p[0] + p[2]) <= cur[0] or (p[1] + p[3]) <= cur[1] or \
                (cur[0] + cur[2]) <= (p[0]) or \
                (cur[1] + cur[3]) <= (p[1]):
            return 0
        #cur in p
        elif p[0] <= cur[0] and p[1] <= cur[1] and (p[0] + p[2]) >= (cur[0] + cur[2]) and (p[1] + p[3]) >= (cur[1] + cur[3]):
            overlap = cur[2] * cur[3]
            iou = overlap / p[2] * p[3]
        #p in cur
        elif cur[0] <= p[0] and cur[1] <= p[1] and (cur[0] + cur[2]) >= (p[0] + p[2]) and (cur[1] + cur[3]) >= (
                        p[1] + p[3]):
            overlap = p[2] * p[3]
            iou = overlap / cur[2] * cur[3]
        #the same x
        elif p[0] == cur[0]:
            x = np.min(cur[2], p[2])
            if p[1] == cur[1]:
                y = np.min(cur[3], p[3])
            elif (p[1] < cur[1] and (p[1] + p[3]) > (cur[1] + cur[3])) or (cur[1] < p[1] and (cur[1] + cur[3]) > (p[1] + p[3])):
                y = np.min(cur[3], p[3])
            else:
                # if p[1] < cur[1]:
                #     y = p[1] + p[3] - cur[1]
                # else:
                #     y = cur[1] + cur[3] - p[1]
                y = np.min(np.abs(cur[1] + cur[3] - p[1]), np.abs(p[1] + p[3] - cur[1]))
            overlap = x * y
            iou = overlap / (p[2] * p[3] + cur[2] * cur[3] - overlap)
        #the same y
        elif p[1] == cur[1]:
            y = np.min(cur[3], p[3])
            if p[0] == cur[0]:
                x = np.min(cur[2], p[2])
            elif (p[0] < cur[0] and (p[0] + p[2]) > (cur[0] + cur[2])) or (cur[0] < p[0] and (cur[0] + cur[2]) > (p[0] + p[2])):
                x = np.min(cur[2], p[2])
            else:
                x = np.min(np.abs(cur[0] + cur[2] - p[0]), np.abs(p[0] + p[2] - cur[0]))
            overlap = x * y
            iou = overlap / (p[2] * p[3] + cur[2] * cur[3] - overlap)
        else:
            x = np.min(np.abs(p[0] + p[2] - cur[0]), np.abs(cur[0] + cur[2] - p[0]))
            y = np.min(np.abs(p[1] + p[3] - cur[1]), np.abs(cur[1] + cur[3] - p[1]))
            overlap = x * y
            iou = overlap / (p[2] * p[3] + cur[2] * cur[3] - overlap)
        return iou


    def product_rule(self, rgb, ir):
        # r1 = np.sort(-rgb, axis=1)
        leng = 10
        idx_r1 = np.argsort(-rgb)
        idx_r2 = idx_r1[0:leng]
        idx_i1 = np.argsort(-ir)
        idx_i2 = idx_i1[0:leng]
        p_r = np.zeros(rgb.shape)
        p_i = np.zeros(ir.shape)
        for i in range(leng):
            p_r[idx_r2[i]] = 1
            p_i[idx_i2[i]] = 1
        pro = rgb * ir
        pro_p = pro * p_r + pro * p_i
        pos = np.unravel_index(np.argmax(pro_p), pro_p.shape)
        if np.isin(pos[0], idx_i2):
            return pos[0], 2
        else:
            return pos[0], 1