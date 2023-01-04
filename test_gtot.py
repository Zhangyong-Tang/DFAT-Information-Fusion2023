# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from DFAT.models.fus2fus import GenerativeNet
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
#from DFAT.models.fus2fus import GenerativeNet
import torch.nn as nn
import time
parser = argparse.ArgumentParser(description='siamrpn tracking')
# parser.add_argument('--dataset', default='OTB100', type=str,
#         help='datasets')
parser.add_argument('--config', default='experiments/siam_base/config.yaml', type=str,
        help='config file')
# parser.add_argument('--snapshot', default='experiments/siam_base/snapshot_refine/checkpoint_e13.pth', type=str,
#         help='snapshot of models to eval')
# parser.add_argument('--video', default='Basketball', type=str,
#         help='eval one special video')
parser.add_argument('--vis', default=True,action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

# load video information
# dataset_root = '/data/Disk_B/zhangyong/tzy/GTOT'
dataset_root = '/data/Disk_D/zhangyong/rgbt/RGB-T234'
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
# cfg.DFNet.DFNet = True
cfg_root = "/data/Disk_D/zhangyong/DFAT/DFAT-19-1/experiments/siam_base/"
# model_file_rgb = join(cfg_root, 'snapshot_refine/checkpoint_e25_rgb.pth')
# model_file = join(cfg_root, 'snapshot_refine/checkpoint_e17.pth')
model_file = '/data/Disk_D/zhangyong/DFAT/DFAT-19-1/experiments/siam_base/snapshot_refine/checkpoint_refine_e18.pth'
# project_root = "/data/Disk_D/zhangyong/DFAT/DFAT-19-1"
# version = 'snapshot'
# checkpoint_num = 'e50'
# model_file = join(project_root, version, 'checkpoint_' + checkpoint_num + '.pth')
# tic = time.time()
model = ModelBuilder()
used = [0]
dist_model = nn.DataParallel(model, used).cuda()
# model_name = args.snapshot.split('/')[-1].split('.')[0]
# load model
# model_file = '/data/Disk_C/tzy_data/snapshot/snapshot_dfnet_0.3701_R-v49/checkpoint_e20.pth'
model = load_pretrain(model, model_file).cuda().eval()
# model_tir = load_pretrain(model, model_file_tir).cuda().eval()
# build tracker
tracker = build_tracker(model)
# toc = time.time()
# print('Setup Tracker casts %f' % (toc-tic))


#for dataset
# fail_path = '/data/Disk_B/zhangyong/siam_motion_test/failure.txt'
# file_root = '/data/Disk_B/zhangyong/siam_motion_test/analysis.txt'
curr_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# with open(fail_path, 'a') as f:
#     f.writelines('\n' + ' ' + curr_time + '\n')
#     f.close()
# cfg.TRACK.PENALTY_K = np.float(0.10)
# cfg.TRACK.WINDOW_INFLUENCE = np.float(0.40)
# cfg.TRACK.LR = np.float(0.30)
fps_total= 0
frame_total = 0

for i in range(num_sequences):
    result = []
    sequence_name = sequence[i].split('\n')[0]
    sequence_path = os.path.join(dataset_root, sequence_name)
    anno_rgb_path = sequence_path + '/visible.txt'
    anno_tir_path = sequence_path + '/infrared.txt'
    rgb_imgs_path = sequence_path + '/visible'
    ir_imgs_path = sequence_path + '/infrared'
    rgb_imgs_files = sorted([f for f in os.listdir(rgb_imgs_path)])
    ir_imgs_files = sorted([f for f in os.listdir(ir_imgs_path)])
    result_file_name = 'DFAT' + '_' + sequence_name + '.txt'
    pred_bboxes = []
    lost_number = 0
    toc = 0
    restart = 0
    frame_counter = 0
    F = 0
    S = 0
    with open(anno_rgb_path) as f:
        ground_truth_rgb_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    with open(anno_tir_path) as f:
        ground_truth_tir_list = [i.split('\n')[0].split(',') for i in f.readlines()]

    op = torch.zeros(len(ground_truth_rgb_list))
    fps_video = 0
    for frame in range(len(ground_truth_rgb_list)):
        tic = time.time()
        # read image
        # print('%s : %d' % (sequence_name, frame + 1))
        im_rgb = cv2.imread(os.path.join(rgb_imgs_path, rgb_imgs_files[frame]))
        im_tir = cv2.imread(os.path.join(ir_imgs_path, ir_imgs_files[frame]))
                        # get gt
                        # groundtruth = [ground_truth_list[frame][]]
                        # pdb.set_trace()
                        # cx, cy, w, h = get_axis_aligned_bbox(np.array(ground_truth_list[frame], dtype='float32'))
                        # gt_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

        im = []
        im = [im_rgb, im_tir]
        if frame == 0 or restart == 1:
            rgb_gt = ground_truth_rgb_list[frame]  # [0].split(',')  # list[list(str)]
            gt_bbox_rgb = [float(rgb_gt[0]), float(rgb_gt[1]), float(rgb_gt[2]),
                           float(rgb_gt[3])]
            tir_gt = ground_truth_tir_list[frame]  # [0].split(',')
            gt_bbox_tir = [float(tir_gt[0]), float(tir_gt[1]), float(tir_gt[2]),
                           float(tir_gt[3])]
            flag = 1
            restart = 0
            gt_init = gt_bbox_rgb
            result.append(gt_init)
            # tic = time.time()
            tracker.init(im, gt_bbox_rgb)
            # print('Initialize Template casts %f' % (time.time() - tic))
            # print("%s, frame:%d initialized" % (sequence_name, frame + 1))
            frame_counter = frame_counter + 1
            if frame_counter >= len(ground_truth_rgb_list):
                break
                            # pred_bboxes.append(1)
        elif frame > 0:
            flag = flag + 1
            # tic = time.time()
            outputs = tracker.track(im)
            # print('Track casts %f' % (time.time() - tic))
            pred_bbox = outputs['bbox']
                            # result.append(pred_bbox)
                            # if frame % 10 == 0 :
                            #     tracker.update(im)
            #print("%s, frame:%d tracked" % (sequence_name, frame + 1))
            if flag % 25 == 0:
                # tic = time.time()
                tracker.update(im, outputs['bbox'])
                # print('Update Template casts %f' % (time.time() - tic))

                            # overlap_rgb = vot_overlap(pred_bbox, gt_bbox_rgb, (im_rgb.shape[1], im_rgb.shape[0]))
                            # overlap_tir = vot_overlap(pred_bbox, gt_bbox_tir, (im_rgb.shape[1], im_rgb.shape[0]))
            pred_bbox = list(map(int, pred_bbox))
            result.append(pred_bbox)
                            # if overlap_rgb > 0:
                            #     # not lost
                            #     result.append(pred_bbox)
                            #     tracker.set_cenetr_old(1)
                            #     print("%s, frame:%d tracked successed" % (sequence_name, frame + 1))
                            # else:
                            #     # lost
                            #     re = result[frame - 1]
                            #     result.append(re)
                            #     tracker.set_cenetr_old(0)
                            #     print("%s, frame:%d tracked failed" % (sequence_name, frame + 1))
        else:
            pred_bboxes.append(0)
        toc = time.time() - tic
        fps_video = fps_video + toc


                    # save result file
                    # if not os.path.exists(save_result_path):

    # f = open(os.path.join(dict_path, result_file_name), "w")
    print('Run video %s used %f seconds, FPS=%f'%(sequence_name, fps_video, len(ground_truth_rgb_list) / fps_video))
    frame_total = frame_total + len(ground_truth_rgb_list)
    fps_total = fps_total + fps_video
    results_path = '/data/Disk_D/zhangyong/DFAT/DFAT-19-1/results_RGBT234'
    file_name = str(cfg.TRACK.PENALTY_K) + '+' + str(cfg.TRACK.WINDOW_INFLUENCE) + '+' + str(cfg.TRACK.LR) + '+e18-101'
    if not os.path.exists(os.path.join(results_path, file_name)):
        os.mkdir(os.path.join(results_path, file_name))
    f = open(os.path.join(results_path, file_name, result_file_name), "w")
    num = len(result)
    for i in range(num):
        tem = result[i]
        tem = str(tem[0]) + ' ' + str(tem[1]) + ' ' + str(tem[0] + tem[2]) + ' ' + str(tem[1]) + ' ' + \
                str(tem[0] + tem[2]) + ' ' + str(tem[1] + tem[3]) + ' ' + str(tem[0]) + ' ' + str(
            tem[1] + tem[3]) + '\n'
        f.writelines(tem)
    f.close()

print('Run dataset %s, FPS=%f'%('rgbt234', frame_total / fps_total))


