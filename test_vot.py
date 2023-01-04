# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import pdb
import cv2
import torch
import numpy as np
#here musr be false ,else Runtime error 11
torch.backends.cudnn.benchmark = False
from os.path import join
from DFAT.core.config import cfg
from DFAT.models.model_builder import ModelBuilder
from DFAT.tracker.tracker_builder import build_tracker
from DFAT.utils.bbox import get_axis_aligned_bbox
from DFAT.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn as nn

parser = argparse.ArgumentParser(description='siamrpn tracking')
# parser.add_argument('--dataset', default='OTB100', type=str,
#         help='datasets')
parser.add_argument('--config', default='experiments/siam_base/config_refine.yaml', type=str,
        help='config file')
# parser.add_argument('--snapshot', default='experiments/siam_base/snapshot_refine/checkpoint_e13.pth', type=str,
#         help='snapshot of models to eval')
# parser.add_argument('--video', default='Basketball', type=str,
#         help='eval one special video')
parser.add_argument('--vis', default='True',action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

# load video information
# dataset_root = '/data/Disk_B/zhangyong/VOT2020/rgbt20/sequences'
dataset_root = '/data/Disk_D/zhangyong/votrgbt2019/sequences'
#video_name = 'humans_corridor_occ_2_A'#
list_path = os.path.join(dataset_root, "list.txt")
sequence_list_file = open(list_path, "r")
sequence =sequence_list_file.readlines()
sequence_list_file.close()
#video_path = os.path.join(dataset_root, video_name)
#anno_path = video_path + '/groundtruth.txt'
#rgb_imgs_path = video_path + '/color/'
#depth_imgs_path = video_path + '/depth/'
#rgb_imgs_files = [f for f in os.listdir(rgb_imgs_path)]
#depth_imgs_files = [f for f in os.listdir(depth_imgs_path)]
num_sequences = len(sequence)

cfg.merge_from_file(args.config)
cur_dir = os.path.dirname(os.path.realpath(__file__))
# create model
model = ModelBuilder()
used = [0]
dist_model = nn.DataParallel(model, used).cuda()
# model_name = args.snapshot.split('/')[-1].split('.')[0]
# load model
project_root = "/data/Disk_D/zhangyong/FusionFromPurpose"
cfg_root = join(project_root, "experiments/siam_base")
version = 'v87'
checkpoint_num = 'checkpoint_e95'
# model_file = join('/data/Disk_C/tzy_space/Trackers/FusionFromPurpose-train2', version, checkpoint_num + '.pth')
#model_file = join('/data/Disk_Z/zhangyong_space/snapshot/ffp', version, checkpoint_num + '.pth')
# model_file = join('J:/Untitled Folder/ffp', version, checkpoint_num + '.pth')
model_file = join("/data/Disk_A/zhangyong/DFAT-19-1/experiments/siam_base", 'snapshot_refine/checkpoint_refine_e18.pth')
model = load_pretrain(model, model_file).cuda().eval()
# model_tir = load_pretrain(model, model_file_tir).cuda().eval()
# build tracker
tracker = build_tracker(model)

sequence_name = 'afterrain'# baby, biketwo, diamond, man4, balancebike, courch, mandrivecar, child1, greyman, mandrivecar
sequence_path = os.path.join(dataset_root, sequence_name)
anno_path = sequence_path + '/groundtruth.txt'
rgb_imgs_path = sequence_path + '/color'
ir_imgs_path = sequence_path + '/ir'
rgb_imgs_files = sorted([f for f in os.listdir(rgb_imgs_path)])
ir_imgs_files = sorted([f for f in os.listdir(ir_imgs_path)])

# writer = SummaryWriter(join('/data/Disk_C/tzy_space/Trackers/FusionFromPurpose/results', version, checkpoint_num, \
#                             sequence_name, 'tir'))
writer = None
# writer = join('/data/Disk_A/tzy_space/Trackers/FusionFromPurpose/results', version, checkpoint_num, \
#                             sequence_name, 'tir')
writer = 0
pred_bboxes = []
    # pre_r = []
    # pre_i = []
lost_number = 0
toc = 0

with open(anno_path) as f:
    ground_truth_list = [i.split('\n')[0].split(',') for i in f.readlines()]

frame_counter = 0
restart = 1
flag = 0
for frame in range(len(ground_truth_list)) :
    tic = cv2.getTickCount()
        #read image
    im_rgb = cv2.imread(os.path.join(rgb_imgs_path, rgb_imgs_files[frame]))
    im_tir = cv2.imread(os.path.join(ir_imgs_path, ir_imgs_files[frame]))
        #get gt
        #groundtruth = [ground_truth_list[frame][]]
    #pdb.set_trace()
    cx, cy, w, h = get_axis_aligned_bbox(np.array(ground_truth_list[frame], dtype='float32'))
    gt_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    # im = []
    im = [im_rgb, im_tir]
    if frame == 0 or restart == 1:
        gt_init = gt_bbox
        tracker.init(im, gt_bbox, writer, frame_counter+1)
        pred_bbox = gt_bbox
        pred_bboxes.append(1)
        restart = 0
        flag = 2
        print('%s : frame %d initialized!' % (sequence_name, frame_counter+1))
    elif frame > 0:
        # outputs = tracker.track(im)
        frame_counter = frame_counter + 1
        outputs = tracker.track(im, frame_counter+1, writer)
        flag = flag + 1
        pred_bbox = outputs['bbox']
        pred_bbox =pred_bbox
        # print('%s : frame %d tracked!' % (sequence_name, frame_counter))
        draw_bb = [int(pred_bbox[i]) for i in range(len(pred_bbox))]
        draw_bb[2] = draw_bb[2] + draw_bb[0]
        draw_bb[3] = draw_bb[3] + draw_bb[1]
        # writer.add_image_with_boxes('result_image_'+ str(frame_counter), im_rgb, draw_bb, dataformats='HWC')
            # pre_r = outputs['bb_r']
            # pre_i = outputs['bb_i']
            # if frame % 10 == 0 :
            #     tracker.update(im_rgb, im_ir, pred_bbox)

        # writer.add_image('a', make_grid())
        overlap = vot_overlap(pred_bbox, gt_bbox, (im_rgb.shape[1], im_rgb.shape[0]))
        print('Predited cx: %f, cy: %f , w: %f, H:%f'%(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]))
        if overlap > 0:
                # not lost
            pred_bboxes.append(pred_bbox)
            print('%s : frame %d tracked! idx: %d' % (sequence_name, frame_counter, outputs['idx']))
            if frame_counter > len(ground_truth_list):
                print('%s : finished!' % (sequence_name))
                break
            # cv2.rectangle(im_rgb, (int(pred_bbox[0]), int(pred_bbox[1])),
            #     (int(pred_bbox[0] + pred_bbox[2]), int(pred_bbox[1] + pred_bbox[3])), (0, 255, 0), 3)#yellow
            # cv2.rectangle(im_rgb, (int(gt_bbox[0]), int(gt_bbox[1])),
            #               (int(gt_bbox[0] + gt_bbox[2]), int(gt_bbox[1] + gt_bbox[3])), (0, 0, 255),3)  # yellow
            # cv2.putText(im_rgb, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            # cv2.imshow(sequence_name, im_rgb)
            # cv2.waitKey(1)
        else:
                # lost object
            pred_bboxes.append(2)
            # frame_counter = frame_counter + 1
            print('%s : frame %d failed!' % (sequence_name, frame_counter))
            frame_counter = frame_counter + 4  # skip 5 frames
            lost_number += 1
            restart = 1
            if frame_counter > len(ground_truth_list):
                print('%s : finished!' % (sequence_name))
                break
    else:
        pred_bboxes.append(0)

print('Totally lost %d times'%(lost_number))

#     toc += cv2.getTickCount() - tic
#     if frame == 0:
#         cv2.destroyAllWindows()
#     if args.vis and frame > 0:
#         gt_bbox = list(map(int, gt_bbox))
#         pred_bbox = list(map(int, pred_bbox))
#         cv2.rectangle(im_rgb, (gt_bbox[0], gt_bbox[1]),
#                         (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)#yellow
#         cv2.rectangle(im_rgb, (pred_bbox[0], pred_bbox[1]),
#                         (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)#blue
#             # cv2.rectangle(im_rgb, (pre_r[0], pre_r[1]),
#             #               (pre_r[0] + pre_r[2], pre_r[1] + pre_r[3]), (255, 0, 0), 3)  # red
#             # cv2.rectangle(im_rgb, (pre_i[0], pre_i[1]),
#             #               (pre_i[0] + pre_i[2], pre_i[1] + pre_i[3]), (0, 0, 255), 3)  # blue
#
#         cv2.putText(im_rgb, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#         cv2.imshow(sequence_name, im_rgb)
#         cv2.waitKey(1)
# toc /= cv2.getTickFrequency()


#for i in range(num_sequences) :
    #sequence_name = sequence[i].split('\n')[0]
    # sequence_name = 'aftertree'
    # sequence_path = os.path.join(dataset_root, sequence_name)
    # anno_path = sequence_path + '/groundtruth.txt'
    # rgb_imgs_path = sequence_path + '/color'
    # ir_imgs_path = sequence_path + '/ir'
    # rgb_imgs_files = [f for f in os.listdir(rgb_imgs_path)]
    # ir_imgs_files = [f for f in os.listdir(ir_imgs_path)]
    #
    # pred_bboxes = []
    # pre_r = []
    # pre_i = []
    # lost_number = 0
    # toc = 0
    #
    # with open(anno_path) as f:
    #     ground_truth_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    #
    #
    # for frame in range(len(ground_truth_list)) :
    #     tic = cv2.getTickCount()
    #     #read image
    #     im_rgb = cv2.imread(os.path.join(rgb_imgs_path, rgb_imgs_files[frame]))
    #     im_ir = cv2.imread(os.path.join(ir_imgs_path, ir_imgs_files[frame]))
    #     #get gt
    #     #groundtruth = [ground_truth_list[frame][]]
    #     cx, cy, w, h = get_axis_aligned_bbox(np.array(ground_truth_list[frame], dtype='float32'))
    #     gt_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    #     if frame == 0 :
    #         gt_init = gt_bbox
    #         tracker.init(im_rgb, im_ir, gt_bbox)
    #         pred_bbox = gt_bbox
    #         pred_bboxes.append(1)
    #     elif frame > 0:
    #         outputs = tracker.track(im_rgb, im_ir)
    #         pred_bbox = outputs['bbox']
    #         # pre_r = outputs['bb_r']
    #         # pre_i = outputs['bb_i']
    #         # if frame % 10 == 0 :
    #         #     tracker.update(im_rgb, im_ir, pred_bbox)
    #
    #
    #         overlap = vot_overlap(pred_bbox, gt_bbox, (im_rgb.shape[1], im_rgb.shape[0]))
    #         if overlap > 0:
    #             # not lost
    #             pred_bboxes.append(pred_bbox)
    #         else:
    #             # lost object
    #             pred_bboxes.append(2)
    #             frame_counter = frame + 5  # skip 5 frames
    #             lost_number += 1
    #     else:
    #         pred_bboxes.append(0)
    #     toc += cv2.getTickCount() - tic
    #     if frame == 0:
    #         cv2.destroyAllWindows()
    #     if args.vis and frame > 0:
    #         '''
    #         cv2.polylines(im_rgb, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
    #                       True, (0, 255, 0), 3)
    #         if cfg.MASK.MASK:
    #             mask = outputs['mask'] > cfg.TRACK.MASK_THERSHOLD
    #             im_rgb[:, :, 2] = mask * 255 + (1 - mask) * im_rgb[:, :, 2]
    #             location_int = np.int0(pred_bbox)
    #             cv2.polylines(im_rgb, [location_int.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
    #             # cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
    #             #          True, (0, 255, 255), 3)
    #         else:
    #             bbox = list(map(int, pred_bbox))
    #             cv2.rectangle(im_rgb, (bbox[0], bbox[1]),
    #                           (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
    #         cv2.putText(im_rgb, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    #         cv2.putText(im_rgb, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #         asr = min(600 / im_rgb.shape[0], 800 / im_rgb.shape[1])
    #         img = cv2.resize(im_rgb, (int(im_rgb.shape[0] * asr), int(im_rgb.shape[1] * asr)))
    #         cv2.imshow(sequence_name, im_rgb)
    #         cv2.waitKey(1)
    #         '''
    #
    #
    #         gt_bbox = list(map(int, gt_bbox))
    #         pred_bbox = list(map(int, pred_bbox))
    #         cv2.rectangle(im_rgb, (gt_bbox[0], gt_bbox[1]),
    #                       (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)#yellow
    #         cv2.rectangle(im_rgb, (pred_bbox[0], pred_bbox[1]),
    #                       (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)#blue
    #         # cv2.rectangle(im_rgb, (pre_r[0], pre_r[1]),
    #         #               (pre_r[0] + pre_r[2], pre_r[1] + pre_r[3]), (255, 0, 0), 3)  # red
    #         # cv2.rectangle(im_rgb, (pre_i[0], pre_i[1]),
    #         #               (pre_i[0] + pre_i[2], pre_i[1] + pre_i[3]), (0, 0, 255), 3)  # blue
    #
    #         cv2.putText(im_rgb, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    #         cv2.imshow(sequence_name, im_rgb)
    #         cv2.waitKey(1)
    # toc /= cv2.getTickFrequency()
    # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
#     #     i + 1, sequence_name, toc, (i + 1) / toc))


