import os
import os.path as osp
import cv2

import sklearn.metrics as mt

def OTSU(filepath,outputpath):

    # 先进行高斯滤波，再使用Otsu阈值法
    for j, pid in enumerate(sorted(os.listdir(filepath))):
        if pid.startswith('.'):
            continue
        pic_list = sorted(os.listdir(osp.join(filepath, pid)), key=lambda x: int(x[:-4]))
        for i, pic_name in enumerate(pic_list):
            img_path = os.path.join(filepath,pid, pic_list[i])
            img = cv2.imread(img_path,0)

            blur = cv2.GaussianBlur(img, (5, 5), 0)
            #blur = cv2.bilateralFilter(img,5,5)
            ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th = cv2.medianBlur(th,5)

            save_dir = osp.join(outputpath, pid)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(osp.join(save_dir, '%s.bmp' % (i+1)), th)
        print('%d/%d'%((j+1),len(os.listdir(filepath))))
        print('saved')
            #imageio.imwrite(os.path.join(img_f_path,'{}.png'.format(i)),slice)

if __name__ == '__main__':
    root_list = ['/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Cirrus/scans',
                 '/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Spectralis/scans',
                 '/media/hxx/9D76-E202/dataset/RETOUCH/RETOUCH-TrainingSet-Topcon/scans']
    for root in root_list:
        OTSU(osp.join(root, 'image'), osp.join(root, 'OTSU_layer'))
        print('dataset #{} has been generated!'.format(root))
