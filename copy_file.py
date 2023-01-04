import os

ori_root = '/data/Disk_D/zhangyong/votrgbt2019/sequences'
tar_path = '/data/Disk_D/zhangyong/ImageFusion/Imagefusion_deepfuse-master/outputs/deepfuse/deepfuse'
cmd = 'cp ' + os.path.join(ori_root, 'list.txt') + ' ' + os.path.join(tar_path, 'list.txt')
os.system(cmd)

def cp_video(video, ori_root, tar_path):
    video_root = os.path.join(ori_root, video)
    i = 1
    for v_root, ds, fs in os.walk(video_root):
        if i == 1:
            for f in fs:
                cmd = 'cp ' + os.path.join(v_root, f) + ' ' + os.path.join(tar_path, video, f)
                print('CP to %s' % (os.path.join(tar_path, video, f)))
                os.system(cmd)
            i = i + 1


def cp_dataset(ori_root, tar_path):
    for root, dirs, files in os.walk(ori_root):
        dirs.sort()
        for video in dirs:
            cp_video(video, ori_root, tar_path)


def main(ori_root, tar_path):
    cp_dataset(ori_root, tar_path)

if __name__ == '__main__':
    main(ori_root, tar_path)