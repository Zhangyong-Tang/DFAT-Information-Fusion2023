# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from random import randint
import random

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(3)

# fus_flag = True
# if fus_flag:
#     fus2fus_model = '/data/Disk_B/zhangyong/siam_motion_test/pretrained_models/fus2fus3.model'
#     fus2fus = GenerativeNet(3,3)
#     # fus2fus = load_pretrain(fus2fus, fus2fus_model)
#     fus2fus.load_state_dict(torch.load(fus2fus_model))
#     fus2fus.cuda()
#
#
# def fusion(x):
#     # rgb = x[0]
#     # tir = x[1]
#
#     Width, Height, C = torch.from_numpy(x[1]).float().size()
#     # rgb = rgb.view(Width, Height, C).detach().cpu().numpy()
#     # tir = tir.view(Width, Height, C).detach().cpu().numpy()
#     # tir = cv2.cvtColor(tir, cv2.COLOR_BGR2GRAY)
#     # rgb_ycrcb = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)
#     # Y2, Cr2, Cb2 = cv2.split(rgb_ycrcb)
#     # Y2 = Y2.reshape(1, 1, Width, Height)
#     rgb = torch.from_numpy(x[0]).float().view(1, C, Width, Height).cuda()
#     tir = torch.from_numpy(x[1]).float().view(1, C, Width, Height).cuda()
#     # out = fus2fus(torch.from_numpy(tir.reshape([1, 1, Width, Height])).float().cuda(), torch.from_numpy(Y2.reshape([1, 1, Width, Height])).float().cuda())
#     out = fus2fus(tir, rgb)
#     # result_Y = (out - torch.min(out)) / (torch.max(out) - torch.min(out) + fus_EPSILON)
#     # result_Y = result_Y * 255
#     # result_Y = result_Y.unit8()
#     # out = torch.stack[result_Y, Cr2, Cb2]
#     l = len(out)
#     temp = []
#     for i in range(l):
#         a = out[i].view(Width, Height, C)
#         temp.append(((a - a.min()) / (a.max() - a.min()) * 255).int().detach().cpu().numpy())
#     return temp


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
dataset_root = '/data/Disk_C/tzy_data/tzy/GTOT'
#video_name = 'humans_corridor_occ_2_A'#
list_path = os.path.join(dataset_root, "gtot.txt")
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
cfg_root = "/data/Disk_B/zhangyong/DFAT-19-1/experiments/siam_base/"
# model_file_rgb = join(cfg_root, 'snapshot_refine/checkpoint_e25_rgb.pth')
# model_file = join(cfg_root, 'snapshot_refine/checkpoint_e17.pth')
# model_file = '/data/Disk_B/zhangyong/single_rpn/pretrained_models/checkpoint_e19-9.1.pth'
# model_file = '/data/Disk_C/tzy_data/snapshot/snapshot_dfnet_0.3984_R-v44/checkpoint_e10.pth'
#model_file = '/data/Disk_C/tzy_data/snapshot/snapshot_dfnet_0.3701_R-v49/checkpoint_e18.pth'
model_file = '/data/Disk_B/zhangyong/DFAT-19-1/experiments/siam_base/snapshot_refine/checkpoint_refine_e18.pth'
model = load_pretrain(model, model_file).cuda().eval()
# model_tir = load_pretrain(model, model_file_tir).cuda().eval()
# build tracker
tracker = build_tracker(model)


#for dataset
# fail_path = '/data/Disk_B/zhangyong/siam_motion_test/failure.txt'
# file_root = '/data/Disk_B/zhangyong/siam_motion_test/analysis.txt'
curr_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# with open(fail_path, 'a') as f:
#     f.writelines('\n' + ' ' + curr_time + '\n')
#     f.close()


PENALTY_K = np.linspace(1, 30, 30) * 0.01 #0.1              0.34
WINDOW_INFLUENCE = np.linspace(10, 45, 36) * 0.01#0.4        0.44
LR = np.linspace(10, 40, 31) * 0.01 # 0.3                    0.3
#TD
#0.1  0.1-0.35/ 0.01
#0.4  0.25-0.5/ 0.01
#0.3  0.15-0.4/ 0.01
#0.31  0.26-0.36/ 0.001
#0.29  0.24-0.34/ 0.001
#0.34  0.29-0.39/ 0.001
#0.293 0.292-0.294 0.0001
#0.267 0.266-0.268 0.0001
#0.333 0.332-0.334 0.0001


#gtot1
#0.01-0.3
#0.10-0.45
#0.10-0.40

#gtot2
#0.2   0.15-0.25
#0.25  0.20-0.30
#0.37  0.32-0.42


np.random.shuffle(PENALTY_K)
np.random.shuffle(WINDOW_INFLUENCE)
np.random.shuffle(LR)
len_p = len(PENALTY_K)
len_w = len(WINDOW_INFLUENCE)
len_l = len(LR)
search_round = 200
count = 0
num_search = len_p * len_w * len_l
select = [randint(0, num_search - 1) + 1 for i in range(search_round)]
for p in range(len_p):
    for w in range(len_w):
        for l in range(len_l):
            count = count + 1
            if count not in select:
                continue
            cfg.TRACK.PENALTY_K = np.float(PENALTY_K[p])
            cfg.TRACK.WINDOW_INFLUENCE = np.float(WINDOW_INFLUENCE[w])
            cfg.TRACK.LR = np.float(LR[l])
            print('Test para, pk=%.4f, wi=%.4f, lr=%.4f\n' % (
            cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR))
            dict_name = 'results_' + str(PENALTY_K[p]) + '_' + str(WINDOW_INFLUENCE[w]) + '_' + str(
                LR[l])
            dict_path = os.path.join('/data/Disk_B/zhangyong/DFAT-19-1/results_GTOT/tir', dict_name)
            if not os.path.exists(dict_path):
                os.mkdir(dict_path)
            for i in range(num_sequences):
                result = []
                sequence_name = sequence[i].split('\n')[0]
                sequence_path = os.path.join(dataset_root, sequence_name)
                anno_rgb_path = sequence_path + '/groundTruth_v.txt'
                anno_tir_path = sequence_path + '/groundTruth_i.txt'
                rgb_imgs_path = sequence_path + '/v'
                ir_imgs_path = sequence_path + '/i'
                rgb_imgs_files = sorted([f for f in os.listdir(rgb_imgs_path)])
                ir_imgs_files = sorted([f for f in os.listdir(ir_imgs_path)])
                result_file_name = 'ours' + '_' + sequence_name + '.txt'
                if not os.path.exists(os.path.join(dict_path, result_file_name)):
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
                    for frame in range(len(ground_truth_rgb_list)):
                        tic = cv2.getTickCount()
                        # read image
                        # print('%s : %d' % (sequence_name, frame + 1))
                        im_rgb = cv2.imread(os.path.join(rgb_imgs_path, rgb_imgs_files[frame]))
                        im_tir = cv2.imread(os.path.join(ir_imgs_path, ir_imgs_files[frame]))
                        # get gt
                        # groundtruth = [ground_truth_list[frame][]]
                        # pdb.set_trace()
                        # cx, cy, w, h = get_axis_aligned_bbox(np.array(ground_truth_list[frame], dtype='float32'))
                        # gt_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                        rgb_gt = ground_truth_rgb_list[frame][0].split(' ')  # list[list(str)]
                        gt_bbox_rgb = [float(rgb_gt[0]), float(rgb_gt[1]), float(rgb_gt[2]) - float(rgb_gt[0]),
                                       float(rgb_gt[3]) - float(rgb_gt[1])]
                        tir_gt = ground_truth_tir_list[frame][0].split(' ')
                        gt_bbox_tir = [float(tir_gt[0]), float(tir_gt[1]), float(tir_gt[2]) - float(tir_gt[0]),
                                       float(tir_gt[3]) - float(tir_gt[1])]

                        im = []
                        #im = [im_rgb, im_tir]
                        im = [im_tir, im_tir]
                        if frame == 0 or restart == 1:
                            flag = 1
                            restart = 0
                            gt_init = gt_bbox_rgb
                            result.append(gt_init)
                            tracker.init(im, gt_bbox_rgb)
                            print("%s, frame:%d initialized" % (sequence_name, frame + 1))
                            frame_counter = frame_counter + 1
                            if frame_counter >= len(ground_truth_rgb_list):
                                break
                            # pred_bboxes.append(1)
                        elif frame > 0:
                            flag = flag + 1
                            outputs = tracker.track(im)
                            pred_bbox = outputs['bbox']
                            # result.append(pred_bbox)
                            # if frame % 10 == 0 :
                            #     tracker.update(im)
                            print("%s, frame:%d tracked" % (sequence_name, frame + 1))
                            #if flag % 10 == 0:
                            #    tracker.update(im, outputs['bbox'])
                            pred_bbox = list(map(int, pred_bbox))

                            # overlap_rgb = vot_overlap(pred_bbox, gt_bbox_rgb, (im_rgb.shape[1], im_rgb.shape[0]))
                            # overlap_tir = vot_overlap(pred_bbox, gt_bbox_tir, (im_rgb.shape[1], im_rgb.shape[0]))
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
                        toc += cv2.getTickCount() - tic

                    # save result file
                    # if not os.path.exists(save_result_path):

                    f = open(os.path.join(dict_path, result_file_name), "w")
                    num = len(result)
                    for i in range(num):
                        tem = result[i]
                        tem = str(tem[0]) + ' ' + str(tem[1]) + ' ' + str(tem[0] + tem[2]) + ' ' + str(tem[1]) + ' ' + \
                              str(tem[0] + tem[2]) + ' ' + str(tem[1] + tem[3]) + ' ' + str(tem[0]) + ' ' + str(
                            tem[1] + tem[3]) + '\n'
                        f.writelines(tem)
                    f.close()
                else:
                    continue




