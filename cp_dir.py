import os
from os.path import join
ori_root = '/data/Disk_D/zhangyong/votrgbt2019/sequences'
tar_path = '/data/Disk_D/zhangyong/ImageFusion/GFF/outputs/GFF'
cmd = 'cp ' + os.path.join(ori_root, 'list.txt') + ' ' + os.path.join(tar_path, 'list.txt')
os.system(cmd)

def cp_dir(old_dir, new_dir):
    cmd = 'cp -r ' + old_dir + ' ' + new_dir
    os.system(cmd)

def cp_mul_dirs(ori_root, dir_root):
    i = 1
    for root, dirs, files in os.walk(dir_root):
        if i == 1:
            for dir in dirs:
                cp = ['ir', 'ir']
                dir_root = join(root, dir)
                ori = join(ori_root, dir)
                cp_dir(join(ori, cp[0]), join(dir_root, cp[1]))
        i = i + 1

def main(ori_path, tar_path):
    cp_mul_dirs(ori_path, tar_path)

if __name__ == '__main__':
    main(ori_root, tar_path)