import cv2
import numpy as np
import os
import os.path as osp
from os.path import exists
import numpy as np
import random

def his_avg(input_path, out_path):
    for j, pid in enumerate(sorted(os.listdir(input_path))):
        print(j)
        if pid.startswith('.'):
            continue
        pic_list = sorted(os.listdir(osp.join(input_path, pid)), key=lambda x: int(x[:-4]))
        for i, pic_name in enumerate(pic_list):

            image = cv2.imread(osp.join(input_path, pid, pic_list[i]))
            (b, g, r) = cv2.split(image)
            bH = cv2.equalizeHist(b)
            gH = cv2.equalizeHist(g)
            rH = cv2.equalizeHist(r)
            result = cv2.merge((bH,gH,rH))
            save_dir = osp.join(out_path, pid)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(osp.join(save_dir, '%d.bmp' % (i + 1)), result)


def main():
    root_list = ['/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Cirrus/scans',
                 '/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Spectralis/scans',
                 '/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Topcon/scans']
    for root in root_list:
        his_avg(osp.join(root, 'trans3channel_images'), osp.join(root, 'his_equa_images'))
        print('dataset #{} has been generated!'.format(root))


if __name__ == '__main__':
    main()
