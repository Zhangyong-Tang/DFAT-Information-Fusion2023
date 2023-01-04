# -*-coding:utf-8-*-
import sys
import cv2
import numpy as np
import os
del os.environ['MKL_NUM_THREADS']
from os.path import join
import torch
from DFAT.core.config import cfg
from DFAT.models.model_builder import ModelBuilder
from DFAT.tracker.tracker_builder import build_tracker
from DFAT.utils.bbox import get_axis_aligned_bbox
from DFAT.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

import vot
# from vot import Rectangle, Polygon, Point

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# modify root
#work_dir = os.path.abspath('/data/Disk_Z/RGBT/')
cfg_root = "/data/Disk_D/zhangyong/DFAT/DFAT-19-1/experiments/siam_base/"
# model_file = join(cfg_root, 'snapshot_refine/checkpoint_refine_e18.pth')
model_file = join(cfg_root, 'snapshot_refine/checkpoint_refine_e18.pth')
# project_root = "/data/Disk_D/zhangyong/DFAT/DFAT-19-1"
# version = 'snapshot'
# checkpoint_num = 'e50'
# model_file = join(project_root, version, 'checkpoint_' + checkpoint_num + '.pth')
cfg_file = join(cfg_root, 'config.yaml')

def warmup(model):
    for i in range(10):
        model.template([torch.FloatTensor(1,3,127,127).cuda(),torch.FloatTensor(1,3,127,127).cuda()])

def setup_tracker():
    cfg.merge_from_file(cfg_file)

    model = ModelBuilder()
    model = load_pretrain(model, model_file).cuda().eval()

    tracker = build_tracker(model)
    warmup(model)
    return tracker



tracker = setup_tracker()

handle = vot.VOT("rectangle", "rgbt")
gt_bbox = handle.region()

rgb_file, ir_file = handle.frame()
#rgb_file = ir_file   #no rgb
#ir_file = rgb_file   #no ir


if not rgb_file or not ir_file:
    sys.exit(0)

#im_rgb = cv2.imread(join(work_dir, image_file))  # HxWxC
#im_ir = cv2.imread(join(work_dir, ir_file))
print(rgb_file)
print(ir_file)
im_rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
im_ir = cv2.imread(ir_file, cv2.IMREAD_COLOR)

gt_bbox_ = [gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]

image = []
# fusion in pixel

image = [im_rgb, im_ir]

# im_ir_h= [cv2.equalizeHist(im_ir[:, :, i]) for i in range(3)]
# cv2.imshow('ir2', np.array(im_ir_h).transpose(1, 2, 0))

tracker.init(image, gt_bbox_)
count = 1

state = gt_bbox
while True:
    # ff += 1
    rgb_file, ir_file= handle.frame()
    #rgb_file = ir_file  #no rgb
    #ir_file = rgb_file   #no ir
      
    if not rgb_file or not ir_file:
        break
    im_rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    im_ir = cv2.imread(ir_file, cv2.IMREAD_COLOR)
    
    image = []
    # fusion in pixel

    image = [im_rgb, im_ir]

    outputs = tracker.track(image)
    count = count + 1
    if count % 15 == 0:
        tracker.update(image, outputs['bbox'])

    
    state = list(map(int, outputs['bbox']))
    state = vot.Rectangle(state[0], state[1], state[2], state[3])
    # report to vot-toolkit
    handle.report(state)








