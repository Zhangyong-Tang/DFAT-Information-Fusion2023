# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
import numbers

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from DFAT.utils.bbox import center2corner, Center
from DFAT.datasets.anchor_target import AnchorTarget
from DFAT.datasets.augmentation import Augmentation
from DFAT.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)   #the number of video sequence
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.mask_format = '{}.{}.m.png'
        self.pick = self.shuffle()
        # self.has_mask = self.name in ['COCO', 'YOUTUBEVOS']
        self.has_mask = False

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w < 2 or h < 2:
                            logger.info('small bb removed, {self.name} {video} {trk} {bbox}'.format(**locals()))
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new



    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            #shuffle
            if cfg.DATASET.SHUFFLE:
                np.random.shuffle(lists)

            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        #import numbers
        if isinstance(frame, numbers.Integral):
            frame = "{:06d}".format(frame)
            image_path = os.path.join(self.root, video,
                                      self.path_format.format(frame, track, 'x'))
            image_anno = self.labels[video][track][frame]
            mask_path = os.path.join(self.root, video, self.mask_format.format(frame, track))
            return image_path, image_anno, mask_path
        else:
            num = len(frame)
            image_path = []
            image_anno = []
            mask_path = []
            for i in range(num):
                f = "{:06d}".format(frame[i])
                image_path.append(os.path.join(self.root, video,
                                          self.path_format.format(f, track, 'x')))
                image_anno.append(self.labels[video][track][f])
                mask_path.append(os.path.join(self.root, video, self.mask_format.format(f, track)))
            return image_path, image_anno, mask_path

    def get_image_anno_rgbt(self, video, track, frame):
        #import numbers
        # if cfg.
        video_name_type = video.split('\\')
        if len(video_name_type) == 2:
            if video_name_type[1] == 'i':
                video_other = video[:-1] + 'v'
            elif video_name_type[1] == 'v':
                video_other = video[:-1] + 'i'
            elif video_name_type[1] == 'visible':
                video_other = video_name_type[0] + '\\infrared'
            else:
                video_other = video_name_type[0] + '\\visible'
        else:
            video_name_type = video_name_type[0].split('/')
            if len(video_name_type) == 2:
                if video_name_type[1] == 'i':
                    video_other = video[:-1] + 'v'
                elif video_name_type[1] == 'v':
                    video_other = video[:-1] + 'i'
                elif video_name_type[1] == 'visible':
                    video_other = video_name_type[0] + '/infrared'
                else:
                    video_other = video_name_type[0] + '/visible'

        if isinstance(frame, numbers.Integral):
            frame = "{:06d}".format(frame)
            image_path = []
            image_path.append(os.path.join(self.root, video, self.path_format.format(frame, track, 'x')))
            image_path.append(os.path.join(self.root, video_other, self.path_format.format(frame, track, 'x')))
            image_anno = []
            image_anno.append(self.labels[video][track][frame])
            image_anno.append(self.labels[video_other][track][frame])
            mask_path = []
            mask_path.append(os.path.join(self.root, video, self.mask_format.format(frame, track)))
            mask_path.append(os.path.join(self.root, video_other, self.mask_format.format(frame, track)))
            return image_path, image_anno, mask_path
        else:
            num = len(frame)
            image_path = []
            image_anno = []
            mask_path = []
            for i in range(num):
                f = "{:06d}".format(frame[i])
                image_path.append(os.path.join(self.root, video,
                                          self.path_format.format(f, track, 'x')))
                image_anno.append(self.labels[video][track][f])
                mask_path.append(os.path.join(self.root, video, self.mask_format.format(f, track)))

            for i in range(num):
                f = "{:06d}".format(frame[i])
                image_path.append(os.path.join(self.root, video_other,
                                          self.path_format.format(f, track, 'x')))
                image_anno.append(self.labels[video_other][track][f])
                mask_path.append(os.path.join(self.root, video_other, self.mask_format.format(f, track)))
            return image_path, image_anno, mask_path

    def get_positive_pair(self, index, dataset):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))  #choose clip???
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        # range_t = int((cfg.TIME.SEARCH_RANGE - 1) / 2)
        # extra_num = int((cfg.TIME.SEARCH_NUM - 1) / 2)
        # if dataset.name in ['VID', 'DET', 'YOUTUBEBB', 'COCO']:
        #     search_before = search_frame
        #     search_after = search_frame
        #     if cfg.TIME.TYPE:
        #         search = [search_before, search_frame, search_after]
        #         search_frame = search
        # else:
        #     search_before = search_frame - (np.random.choice(range_t, extra_num) + 1)  # not the same
        #     search_after = search_frame + (np.random.choice(range_t, extra_num) + 1)
            # if cfg.TIME.TYPE:
            #     search = []
            #     for i in range(extra_num):
            #         search.append(search_before[i])
            #     search.append(search_frame)
            #     for i in range(extra_num):
            #         search.append(search_after[i])
            #     search_frame = search

        return self.get_image_anno_rgbt(video_name, track, template_frame), \
               self.get_image_anno_rgbt(video_name, track, search_frame), video_name
    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self):
        super(TrkDataset, self).__init__()

        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create anchor target
        if cfg.ANCHOR.TYPE:
            self.anchor_target = AnchorTarget()
        else:
            a = 0

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                name,
                subdata_cfg.ROOT,
                subdata_cfg.ANNO,
                subdata_cfg.FRAME_RANGE,
                subdata_cfg.NUM_USE,
                # start_rgb
                start
            )
            # start_rgb += sub_dataset.num
            start += sub_dataset.num
            # self.num_rgb += sub_dataset.num_use
            self.num += sub_dataset.num_use

            sub_dataset.log()
            # self.rgb_dataset.append(sub_dataset)
            self.all_dataset.append(sub_dataset)


        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = int(videos_per_epoch / 2 if videos_per_epoch > 0 else self.num)
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            #shuffle
            if cfg.DATASET.SHUFFLE:
                np.random.shuffle(p)

            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]



    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        for kk in range(100):
            index = index + kk
            try:
                index = self.pick[index]
                dataset, index = self._find_dataset(index)

                gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
                neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

                # get one dataset
                if neg:
                    template = dataset.get_random_target(index)
                    search = np.random.choice(self.all_dataset).get_random_target()
                else:
                    index = index if (index % 2) == 1 else (index + 1)
                    template, search, video_name = dataset.get_positive_pair(index, dataset)



                template_img = [cv2.imread(template[0][0].replace('\\', '/')), cv2.imread(template[0][1].replace('\\', '/'))]
                search_img = [cv2.imread(search[0][0].replace('\\', '/')), cv2.imread(search[0][1].replace('\\', '/'))]
                template_box = list(self._get_bbox(template_img[i], template[1][i]) for i in range(len(template_img)))
                search_box = list(self._get_bbox(search_img[i], search[1][i]) for i in range(len(search_img)))
                search_mask = np.zeros(search_img[0].shape[:2], dtype=np.float32)
                template_mask = np.zeros(template_img[0].shape[:2], dtype=np.float32)

                num = len(template_img) / 2

                template = []
                bbox_ = []
                mask_t = []
                for i in range(len(template_img)):
                    template_t, bbox_t, mask_tt = self.template_aug(template_img[i],
                                                                    template_box[i],
                                                                    cfg.TRAIN.EXEMPLAR_SIZE,
                                                                    gray=gray, mask=template_mask)
                    template.append(template_t.transpose((2, 0, 1)).astype(np.float32))
                    bbox_.append(bbox_t)
                    mask_t.append(mask_tt)

                search = []
                bbox = []
                mask = []
                for i in range(len(search_img)):
                    search1, bbox1, mask1 = self.search_aug(search_img[i],
                                                            search_box[i],
                                                            cfg.TRAIN.SEARCH_SIZE,
                                                            gray=gray, mask=search_mask)
                    search.append(search1.transpose((2, 0, 1)).astype(np.float32))
                    bbox.append(bbox1)
                    mask.append(mask1)

                delta_weight = np.zeros([5, 25, 25], dtype=np.float32)
                cls = []
                delta = []
                delta_weight = []
                overlap = []
                for i in range(len(search_img)):
                    cls1, delta1, delta_weight1, overlap1 = self.anchor_target(bbox[i], cfg.TRAIN.OUTPUT_SIZE, neg)
                    cls.append(cls1)
                    delta.append(delta1)
                    overlap.append(overlap1)
                    delta_weight.append(delta_weight1)
                # get labels
                # cls, delta, delta_weight, overlap = self.anchor_target(
                #     bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
                mask_weight = np.zeros([1, cls[0].shape[0], cls[0].shape[1]], dtype=np.float32)

                mask = (np.expand_dims(mask, axis=0) > 0.5) * 2 - 1

                return {
                    'template': template,
                    'search': search,
                    'label_cls': cls,
                    'label_loc': delta,
                    'video_index': torch.ones(1) * index,
                    'label_loc_weight': delta_weight,
                    'bbox': list(np.array(bbox[i], np.float32) for i in range(int(num) * 2)),
                    'label_mask': np.array(mask, np.float32),
                    'label_mask_weight': np.array(mask_weight, np.float32),
                    'mask_template': np.array(mask_t, np.float32)
                }

                break
            except:
                a = 100
