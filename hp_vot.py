# from __future__ import division
# from __future__ import print_function

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import numpy as np
import torch
import random
from random import randint
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from DFAT.models.model_builder import ModelBuilder
from DFAT.tracker.tracker_builder import build_tracker
from DFAT.utils.bbox import get_axis_aligned_bbox
from DFAT.utils.model_load import load_pretrain
from DFAT.core.config import cfg

torch.set_num_threads(1)

restart_flag = 1
def parse_range(range_str):
    param = map(float, range_str.split(','))
    return np.arange(*param)


def parse_range_int(range_str):
    param = map(int, range_str.split(','))
    return np.arange(*param)

##########   For ALL
#####    Original Settings
####         PENALTY_K: 0.10
####         WINDOW_INFLUENCE: 0.40
####         LR: 0.30
####         Temporature: 1.0

#####    1st round hp sraech (start, end, step)
####         PENALTY_K: '0.01, 0.5, 0.05'
####         WINDOW_INFLUENCE: '0.1, 0.7, 0.05'
####         LR: '0.1, 0.5, 0.05'
####         Temporature: '1.0, 1.1, 0.2'   #invalid
####         Times: 1000
#############0.46-0.40-0.35-1.0

#####    2st round hp sraech (start, end, step)
####         PENALTY_K: '0.41, 0.50, 0.01'
####         WINDOW_INFLUENCE: '0.35, 0.45, 0.01'
####         LR: '0.30, 0.40, 0.01'
####         Temporature: '1.0, 1.1, 0.2'   #invalid
####         Times: 1000


##########   For ALL   topk=3
#####    Original Settings
####         PENALTY_K: 0.10
####         WINDOW_INFLUENCE: 0.40
####         LR: 0.30
####         Temporature: 1.0

#####    1st round hp sraech (start, end, step)
####         PENALTY_K: '0.05, 0.15, 0.01'
####         WINDOW_INFLUENCE: '0.35, 0.45, 0.01'
####         LR: '0.25, 0.35, 0.01'
####         Temporature: '1.0, 1.1, 0.2'   #invalid
####         Times: 1000
#############0.46-0.40-0.35-1.0

#####    2st round hp sraech (start, end, step)
####         PENALTY_K: '0.41, 0.50, 0.01'
####         WINDOW_INFLUENCE: '0.35, 0.45, 0.01'
####         LR: '0.30, 0.40, 0.01'
####         Temporature: '1.0, 1.1, 0.2'   #invalid
####         Times: 1000

parser = argparse.ArgumentParser(description='Hyperparamter search')
#TFS (train from scratch) + Trans (transformer) + SR (selective refinement)
#"/data/Disk_A/tzy_space/Trackers/FusionFromPurpose-train2/v78/checkpoint_e151.pth"      TFS + Trans + SR : block = 3
parser.add_argument('--snapshot', default="/data/Disk_D/zhangyong/DFAT/DFAT-19-1/checkpoint_e50.pth", type=str, help='snapshot of model')
# '/data/Disk_B/zhangyong/siam_motion_test/experiments/siam_base/snapshot_refine/checkpoint_e16.pth'
# '/data/Disk_C/tzy_data/snapshot/snapshot_motion_0.3701_LG-7.72/checkpoint_e14.pth'
# '/data/Disk_C/tzy_data/snapshot/snapshot_motion_0.3701_LG-7.70-1/'
parser.add_argument('--dataset', default='VOTRGBT2019', type=str, help='dataset name to eval')
parser.add_argument('--penalty-k', default='0.115, 0.125, 0.001', type=parse_range)
parser.add_argument('--window-influence', default='0.385, 0.395, 0.001', type=parse_range)
parser.add_argument('--lr', default='0.285, 0.295, 0.001', type=parse_range)
parser.add_argument('--Temporature', default='1.0, 1.1, 0.2', type=parse_range)
parser.add_argument('--search-region', default='255,256,8', type=parse_range_int)
parser.add_argument('--config', default="/data/Disk_D/zhangyong/DFAT/DFAT-19-1/experiments/siam_base/config.yaml", type=str)
args = parser.parse_args()

writer = None
def get_bbox(bbox):
    if len(bbox) == 8:
        gt = []
        x = bbox[0::2]
        y = bbox[1::2]
        x_max = max(x)
        y_max = max(y)
        x_min = min(x)
        y_min = min(y)
        gt.append(x_min)#get left top
        gt.append(y_min)
        gt.append(x_max-x_min)#get w and h
        gt.append(y_max-y_min)
        return gt
    else:
        return bbox
def run_tracker(tracker, img, gt, video_name, restart=True):
    frame_counter = 0
    lost_number = 0
    toc = 0
    pred_bboxes = []
    if restart:  # VOT2016 and VOT 2018 or VOTRGBT2019
        for idx, (img, gt_bbox_) in enumerate(video):
            if len(gt_bbox_) == 4:  #eight for votrgbt2019
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                           gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                           gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                           gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
            tic = cv2.getTickCount()
            # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            # gt_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            # gt_bbox = [float(gt_bbox[0]), float(gt_bbox[1]), float(gt_bbox[2]) - float(gt_bbox[0]), float(gt_bbox[5]) - float(gt_bbox[1])]
            gt_bbox = get_bbox(gt_bbox_)
            if idx == frame_counter:
                print('%s-%d:%s,%s,%s,%s'%(video_name,idx,gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3]))
                tracker.init(img, gt_bbox)
                flag = 1
                pred_bbox = gt_bbox
                pred_bboxes.append(1)
            elif idx > frame_counter:
                flag = flag + 1
                outputs = tracker.track(img)
                # print('%d'%(flag))
                pred_bbox_ = outputs['bbox']
                pred_bbox = list(map(float, pred_bbox_))
                # print('%f, %f, %f, %f' % (pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]))
                overlap = vot_overlap(pred_bbox, gt_bbox_,
                                      (img[0].shape[1], img[0].shape[0]))
                if overlap > 0:
                    # not lost
                    # pred_bbox = list(map(int, pred_bbox))
                    pred_bboxes.append(pred_bbox)
                else:
                    # lost object
                    pred_bboxes.append(2)
                    frame_counter = idx + 5  # skip 5 frames
                    lost_number += 1
            else:
                pred_bboxes.append(0)
            toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        print('Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
            video_name, toc, idx / toc, lost_number))
        return pred_bboxes
    else:
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
        toc /= cv2.getTickFrequency()
        print('Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            video_name, toc, idx / toc))
        return pred_bboxes, scores, track_times

def _check_and_occupation(video_path, result_path, version_path):
    #if os.path.isdir(version_path):
    #    return True, 1
    if os.path.isfile(result_path):
        return True, 0
    try:
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
    except OSError as err:
        print(err)

    with open(result_path, 'w') as f:
        f.write('Occ')
    return False, 0

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    seed_torch(1234567)#1234567

    num_search = len(args.penalty_k) \
                 * len(args.window_influence) \
                 * len(args.lr) \
                 * len(args.search_region) \
                 * len(args.Temporature)
    print("Total search number: {}".format(num_search))


    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    dataset_root = '/data/Disk_D/zhangyong/votrgbt2019/sequences'
    # create dataset
    # get the gt and file_root
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)


    def warmup(model):
        for i in range(10):
            model.template([torch.FloatTensor(1, 3, 127, 127).cuda(), torch.FloatTensor(1, 3, 127, 127).cuda()])

    warmup(model)

    result_root = '/data/Disk_D/zhangyong/DFAT/DFAT-19-1/votrgbt192'
    model_name = args.snapshot.split('/')[-1].split('.')[0]
    benchmark_path = os.path.join('hp_search_result', args.dataset)
    seqs = list(range(len(dataset)))

    np.random.shuffle(args.penalty_k)
    np.random.shuffle(args.window_influence)
    np.random.shuffle(args.lr)
    np.random.shuffle(args.Temporature)
    search_round = 1000
    search_round = min(search_round, num_search)
    select = [randint(0, num_search-1) + 1 for i in range(search_round)]
    count = 0
    flag_count = 0
    for pk in args.penalty_k:
        for wi in args.window_influence:
            for lr in args.lr:
                for Temporature in args.Temporature:
                    for ins in args.search_region:
                        # pk = 0.10
                        # wi = 0.40
                        # lr = 0.30
                        flag_count = flag_count + 1
                        if flag_count not in select:
                            continue
                        cfg.TRACK.PENALTY_K = float(pk)
                        cfg.TRACK.WINDOW_INFLUENCE = float(wi)
                        cfg.TRACK.LR = float(lr)
                        cfg.TRACK.Temporature = float(Temporature)
                        ins = int(ins)
                        # rebuild tracker
                        tracker = build_tracker(model)
                        tracker_path = os.path.join(benchmark_path,
                                                    (model_name +
                                                     '_r{}'.format(ins) +
                                                     '_pk-{:.4f}'.format(pk) +
                                                     '_wi-{:.4f}'.format(wi) +
                                                     '_lr-{:.4f}'.format(lr) +
                                                     '_tem-{:.4f}'.format(Temporature)))
                        print('Test para, pk=%.4f, wi=%.4f, lr=%.4f, tem=%.4f\n' % (pk, wi, lr, Temporature))
                        is_version_finished_a = 0
                        for idx in seqs:
                            # idx = 4
                            video = dataset[idx]
                            # load image
                            video.load_img()
                            #if is_version_finished_a:
                            #    break
                            if 'VOT2016' == args.dataset or 'VOT2018' == args.dataset:
                                video_path = os.path.join(tracker_path, 'baseline', video.name)
                                result_path = os.path.join(video_path, video.name + '_001.txt')
                                if _check_and_occupation(video_path, result_path):
                                    continue
                                pred_bboxes = run_tracker(tracker, video.imgs,
                                                          video.gt_traj, video.name, restart=True)
                                with open(result_path, 'w') as f:
                                    for x in pred_bboxes:
                                        if isinstance(x, int):
                                            f.write("{:d}\n".format(x))
                                        else:
                                            f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
                            elif 'VOTRGBT2019' == args.dataset:
                                video_path = os.path.join(result_root, tracker_path, 'baseline', video.name)
                                result_path = os.path.join(video_path, video.name + '_001.txt')
                                version_path = os.path.join(result_root, tracker_path)
                                is_file_occupied, is_version_finished = _check_and_occupation(video_path, result_path, version_path)
                                is_version_finished_a = is_version_finished
                                if is_file_occupied:
                                    continue
                                pred_bboxes = run_tracker(tracker, video.imgs,
                                                          video.gt_traj, video.name, restart=True)
                                with open(result_path, 'w') as f:
                                    for x in pred_bboxes:
                                        if isinstance(x, int):
                                            f.write("{:d}\n".format(x))
                                        else:
                                            f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
                            elif 'VOT2018-LT' == args.dataset:
                                video_path = os.path.join(tracker_path, 'longterm', video.name)
                                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                                if _check_and_occupation(video_path, result_path):
                                    continue
                                pred_bboxes, scores, track_times = run_tracker(tracker,
                                                                               video.imgs, video.gt_traj, video.name,
                                                                               restart=False)
                                pred_bboxes[0] = [0]
                                with open(result_path, 'w') as f:
                                    for x in pred_bboxes:
                                        f.write(','.join([str(i) for i in x]) + '\n')
                                result_path = os.path.join(video_path,
                                                           '{}_001_confidence.value'.format(video.name))
                                with open(result_path, 'w') as f:
                                    for x in scores:
                                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                                result_path = os.path.join(video_path,
                                                           '{}_time.txt'.format(video.name))
                                with open(result_path, 'w') as f:
                                    for x in track_times:
                                        f.write("{:.6f}\n".format(x))
                            elif 'GOT-10k' == args.dataset:
                                video_path = os.path.join('epoch_result', tracker_path, video.name)
                                if not os.path.isdir(video_path):
                                    os.makedirs(video_path)
                                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                                with open(result_path, 'w') as f:
                                    for x in pred_bboxes:
                                        f.write(','.join([str(i) for i in x]) + '\n')
                                result_path = os.path.join(video_path,
                                                           '{}_time.txt'.format(video.name))
                                with open(result_path, 'w') as f:
                                    for x in track_times:
                                        f.write("{:.6f}\n".format(x))
                            else:
                                result_path = os.path.join(tracker_path, '{}.txt'.format(video.name))
                                if _check_and_occupation(tracker_path, result_path):
                                    continue
                                pred_bboxes, _, _ = run_tracker(tracker, video.imgs,
                                                                video.gt_traj, video.name, restart=False)
                                with open(result_path, 'w') as f:
                                    for x in pred_bboxes:
                                        f.write(','.join([str(i) for i in x]) + '\n')
                        # free img
                        video.free_img()
                        count = count + 1
                        if count >= search_round:
                            break
                if count >= search_round:
                    break
            if count >= search_round:
                break
        if count >= search_round:
            break

