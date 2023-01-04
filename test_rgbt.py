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
torch.backends.cudnn.benchmark = True
from DFAT.core.config import cfg
from DFAT.models.model_builder import ModelBuilder
from DFAT.tracker.tracker_builder import build_tracker
from DFAT.utils.bbox import get_axis_aligned_bbox
from DFAT.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='siamrpn tracking')
# parser.add_argument('--dataset', default='OTB100', type=str,
#         help='datasets')
parser.add_argument('--config', default='experiments/siam_base/config-pixel.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='snapshot_refine/checkpoint_refine_e18.pth', type=str,
        help='snapshot of models to eval')
# parser.add_argument('--video', default='Basketball', type=str,
#         help='eval one special video')
parser.add_argument('--vis', default='True',action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

# load video information
dataset_root = '/data/Disk_D/zhangyong/votrgbt2019/sequences'
real_root = '/data/Disk_D/zhangyong/ImageFusion/GFF/outputs/GFF'
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
cfg_root = "/data/Disk_D/zhangyong/DFAT/DFAT-19-1/experiments/siam_base/"
# model_file = join(cfg_root, 'snapshot_refine/checkpoint_refine_e18.pth')
model_file = os.path.join(cfg_root, 'snapshot_refine/checkpoint_refine_e18.pth')
# load model
model = load_pretrain(model, model_file).cuda().eval()
# build tracker
tracker = build_tracker(model)

sequence_name = 'afterrain'
sequence_path = os.path.join(dataset_root, sequence_name)
anno_path = sequence_path + '/groundtruth.txt'
rgb_imgs_path = os.path.join(real_root, sequence_name) + '/color'
ir_imgs_path = os.path.join(real_root, sequence_name) + '/ir'
rgb_imgs_files = [f for f in os.listdir(rgb_imgs_path)]
ir_imgs_files = [f for f in os.listdir(ir_imgs_path)]

pred_bboxes = []
    # pre_r = []
    # pre_i = []
lost_number = 0
toc = 0

with open(anno_path) as f:
    ground_truth_list = [i.split('\n')[0].split(',') for i in f.readlines()]


for frame in range(len(ground_truth_list)) :
    tic = cv2.getTickCount()
        #read image
    im_rgb = cv2.imread(os.path.join(rgb_imgs_path, rgb_imgs_files[frame]))
    im_ir = cv2.imread(os.path.join(ir_imgs_path, ir_imgs_files[frame]))
        #get gt
        #groundtruth = [ground_truth_list[frame][]]
    #pdb.set_trace()
    cx, cy, w, h = get_axis_aligned_bbox(np.array(ground_truth_list[frame], dtype='float32'))
    gt_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    if frame == 0 :
        gt_init = gt_bbox
        tracker.init(im_rgb, im_ir, gt_bbox)
        pred_bbox = gt_bbox
        pred_bboxes.append(1)
    elif frame > 0:
        outputs = tracker.track(im_rgb, im_ir)
        pred_bbox = outputs['bbox']
            # pre_r = outputs['bb_r']
            # pre_i = outputs['bb_i']
            # if frame % 10 == 0 :
            #     tracker.update(im_rgb, im_ir, pred_bbox)


        overlap = vot_overlap(pred_bbox, gt_bbox, (im_rgb.shape[1], im_rgb.shape[0]))
        if overlap > 0:
                # not lost
            pred_bboxes.append(pred_bbox)
        else:
                # lost object
            pred_bboxes.append(2)
            frame_counter = frame + 5  # skip 5 frames
            lost_number += 1
    else:
        pred_bboxes.append(0)
    toc += cv2.getTickCount() - tic
    if frame == 0:
        cv2.destroyAllWindows()
    if args.vis and frame > 0:
        gt_bbox = list(map(int, gt_bbox))
        pred_bbox = list(map(int, pred_bbox))
        cv2.rectangle(im_rgb, (gt_bbox[0], gt_bbox[1]),
                        (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)#yellow
        cv2.rectangle(im_rgb, (pred_bbox[0], pred_bbox[1]),
                        (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)#blue
            # cv2.rectangle(im_rgb, (pre_r[0], pre_r[1]),
            #               (pre_r[0] + pre_r[2], pre_r[1] + pre_r[3]), (255, 0, 0), 3)  # red
            # cv2.rectangle(im_rgb, (pre_i[0], pre_i[1]),
            #               (pre_i[0] + pre_i[2], pre_i[1] + pre_i[3]), (0, 0, 255), 3)  # blue

        cv2.putText(im_rgb, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(sequence_name, im_rgb)
        cv2.waitKey(1)
toc /= cv2.getTickFrequency()




'''
# load ground_truth
with open(anno_path) as f:
    ground_truth_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    ground_truth = [(float(i[0]), float(i[1]), float(i[2]), float(i[3])) for i in ground_truth_list]

# ground_truth = torch.from_numpy(np.array(ground_truth, dtype='float32'))
ground_truth = np.array(ground_truth, dtype='float32')
gt_bbox_init = ground_truth[0, :]

cfg.merge_from_file(args.config)

model = ModelBuilder()

# load model
model = load_pretrain(model, args.snapshot).cuda().eval()

# build tracker
tracker = build_tracker(model)

model_name = args.snapshot.split('/')[-1].split('.')[0]


pred_bboxes = []
scores = []
track_times = []
frame = 0

for frame in range(num_frames):


    im_path = os.path.join(rgb_imgs_path, rgb_imgs_files[frame])
    depth_path = os.path.join(depth_imgs_path, depth_imgs_files[frame])
    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, -1)
    # cv2.namedWindow('Image')
    # cv2.imshow('Image', im)
    # cv2.waitKey(0)
    # cv2.destroyWindows()

    if frame == 0:
        cx, cy, w, h = get_axis_aligned_bbox(gt_bbox_init)
        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        tracker.init(im, gt_bbox_)
        pred_bbox = gt_bbox_
        scores.append(None)
        pred_bboxes.append(pred_bbox)

        # calculate depth histogram
        depth_target = tracker.get_patch(depth, np.array(pred_bbox[0:2]), np.array(pred_bbox[2:]))
        # calculate histogram
        depth_hist = tracker.get_hist(depth_target)
        init_depth_hist = depth_hist
        # plt.imshow(depth_target, cmap='gray', vmin=0, vmax=8000)
        # plt.show()
    else:
        outputs = tracker.track(im)
        pred_bbox = outputs['bbox']
        # pred_bboxes.append(pred_bbox)
        # scores.append(outputs['best_score'])

        # depth_hist_pre = depth_hist
        depth_target = tracker.get_patch(depth, np.array(pred_bbox[0:2]), np.array(pred_bbox[2:]))
        depth_hist = tracker.get_hist(depth_target)
        similarity = np.sum(np.sqrt(depth_hist * init_depth_hist))

        if ((outputs['best_score'] < 0.6) and (similarity < 0.6)) or (outputs['best_score'] < 0.1):
        # if outputs['best_score'] < 0.1:
            re_detection = 1
        else:
            re_detection = 0
            init_depth_hist = depth_hist

        if re_detection == 1:
            re_outputs = tracker.redetection(im, depth, init_depth_hist)

            if (re_outputs['best_score'] != 0):

                best_score = re_outputs['best_score']
                pred_bbox = re_outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(best_score)

            else:
                best_score = re_outputs['best_score']
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(best_score)
        else:
            pred_bboxes.append(pred_bbox)
            scores.append(outputs['best_score'])


        # print(similarity, re_detection)
        # print(scores[frame])
    if frame == 0:
        cv2.destroyAllWindows()
    if args.vis and frame > 0:

        if math.isnan(ground_truth[frame, 1]) != True:
            gt_bbox = list(map(int, ground_truth[frame,:]))
            cv2.rectangle(im, (gt_bbox[0], gt_bbox[1]),
                          (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 0, 255), 3)
        if not pred_bbox[1] is None:
            pred_bbox = list(map(int, pred_bbox))
            cv2.rectangle(im, (pred_bbox[0], pred_bbox[1]),
                          (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
        cv2.putText(im, str(frame), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(im, str(scores[frame]), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(im, str(similarity), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(video_name, im)
        cv2.waitKey(1)
'''
