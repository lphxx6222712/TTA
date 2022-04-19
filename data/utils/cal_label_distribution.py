import cv2
import numpy as np
import os
import os.path as osp
from os.path import exists
import numpy as np
import random
import csv

def gen_point_label(img_path, image_path):
    domain = img_path.split('/')[-3].split('-')[-1]
    output_path = osp.join(image_path, domain)
    #if not osp.exists(output_path):
    #os.makedirs(output_path)
    filepath = osp.join(output_path, 'distribution.csv')
    header = ['pid', 'slice', 'background', 'PED', 'SRF', 'IRF']
    with open(filepath,'w') as f:

        writer = csv.writer(f)
        writer.writerow(header)
        for j, pid in enumerate(sorted(os.listdir(img_path))):
            print(j)
            if pid.startswith('.'):
                continue
            pic_list = sorted(os.listdir(osp.join(img_path, pid)), key=lambda x: int(x[:-4]))
            for i, pic_name in enumerate(pic_list):
                # 191=SRF 255=IRF, 128 = PED
                label = cv2.imread(osp.join(img_path, pid, pic_list[i]),0)
                background = len(np.where(label==0)[0])
                PED = len(np.where(label==128)[0])
                SRF = len(np.where(label==191)[0])
                IRF = len(np.where(label==255)[0])
                data =[pid,i,background,PED, SRF, IRF]
                writer.writerow(data)



def main():
    root_list = ['/home/isalab201/Documents/hxx/dataset/RETOUCH/RETOUCH-TrainingSet-Cirrus/scans',
                 '/home/isalab201/Documents/hxx/dataset/RETOUCH/RETOUCH-TrainingSet-Spectralis/scans',
                 '/home/isalab201/Documents/hxx/dataset/RETOUCH/RETOUCH-TrainingSet-Topcon/scans']
    for root in root_list:
    #root = '/home/isalab201/Documents/hxx/dataset/RETOUCH/RETOUCH-TrainingSet-Cirrus/scans'
        gen_point_label(osp.join(root, 'label'), osp.join(root, 'distribution'))
        print('dataset #{} has been generated!'.format(root))


if __name__ == '__main__':
    main()
