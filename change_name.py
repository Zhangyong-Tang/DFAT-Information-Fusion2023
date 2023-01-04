import os

file_root = '/data/Disk_D/zhangyong/ImageFusion/GFF/outputs/GFF'


def list2str(l):
    s = ''
    for i in range(len(l)):
        s = s + '/' + l[i] if i!=0 else s + l[i]
def change_file(file_path, video_index):
    dir_path = os.path.join(file_path, video_index)

    for root, dirs, files in os.walk(dir_path):
        for f in files:
            # p = f.split('\\')
            # p[-1] = p[-1].replace('v','i')
            # save_path = list2str(p)
            save_path = os.path.join(root, f.replace('.jpg', 'v.jpg'))
            print('CN to %s' % (save_path))
            cmd = 'mv ' + os.path.join(root, f) + ' ' + save_path
            os.system(cmd)


def change_dir(dir_root):
    change = ['ir', 'ir_ori']
    cmd = 'mv ' + os.path.join(dir_root, change[0]) + ' ' + os.path.join(dir_root, change[1])
    print('CN for %s' % (dir_root))
    os.system(cmd)


def change_dir2file(dir_root):
    change_file(dir_root, 'fuse_unifusion')

def change_mul_dir(file_root):
    i = 1
    for root, dirs, files in os.walk(file_root):  ###   dirs === videos
        if i == 1:
            for dir in dirs:
                change_dir(os.path.join(root, dir))
        i = i + 1


def main(file_root):
    # video_index = 'woman89/ir'
    # change_file(file_root, video_index)
    change_mul_dir(file_root)

if __name__ == '__main__':
    main(file_root)