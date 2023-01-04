from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import datetime
from tools.eval import main1
import torch
import random
import numpy as np
tracker_name = ['dfat']

# tracker_dir = ['/data/Disk_B/zhangyong/vot-matlab/vot-toolkit/siam_motion/0.3984_motion_7.36-e17',\
#              '/data/Disk_B/zhangyong/vot-matlab/vot-toolkit/siam_motion/0.3990_dfnet_v24-e16']
dataset = 'VOTRGBT2019'
tracker_dir = '/data/Disk_D/zhangyong/DFAT/DFAT-19-1/votrgbt192/hp_search_result'
eao = []
ac = []
ro = []
ln = []
name = []
curr_time = datetime.datetime.now()
time_str = datetime.datetime.strftime(curr_time,'%Y-%m-%d %H:%M:%S')
result_path = os.path.join(tracker_dir, 'result')
with open(result_path, 'a') as f:
    f.writelines(time_str + '\n' + '----------------------------\n' + '----------------------------\n')
    f.close()
def main(tracker_dir):
        for root, dir, files in os.walk(os.path.join(tracker_dir, dataset)):
            dir.sort()
            for i in range(len(dir)):
                print('%s' % (dir[i]))
                try:
                    result = main1(os.path.join(tracker_dir, dataset), [dir[i]])
                except:
                    continue
                eao.append(result['eao'])
                ac.append(result['accuracy'])
                ro.append(result['robustness'])
                ln.append(result['lostnumber'])
                name.append(result['name'])
                line_str = str(result['name']) + ': ac-' + str(result['accuracy']) + ' ro-' + str(result['robustness']) + ' ln-' \
                           + str(result['lostnumber']) + ' eao-' + str(result['eao']) + '\n'
                with open(result_path, 'a') as f:
                    f.writelines(line_str)
                    f.close()



    # main1(tracker_dir, name)

# def seed_torch(seed=0):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # seed_torch(123456)
    main(tracker_dir)


    # main()