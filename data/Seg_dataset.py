from torch.utils.data import Dataset
from os.path import join, exists
from PIL import Image
import torch
import os
import numpy as np
import torchvision.transforms as tt
import data.seg_transforms as st
import PIL
import random
import cv2


class SegList(Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir, device, hist_equal):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.hist_image_list = None
        self.hist_label_list = None
        self.layer_list = None
        self.bbox_list = None
        self.hist_equal = hist_equal
        self.read_lists(device, hist_equal)
    def __getitem__(self, index):
        if self.phase == 'train':
            data = [Image.open(join(self.data_dir, self.image_list[index]))]
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
            data = list(self.transforms(*data))

            layer = cv2.resize(cv2.imread(join(self.data_dir, self.layer_list[index]),0),(496,496))
            layer = torch.from_numpy(layer)/255
            p = os.path.split(self.label_list[index])
            file_name = p[-1]
            patient = os.path.split(p[0])[-1]
            id = int(file_name.split('.', 1)[0])

            data = [data[0], data[1].long(), layer.long()]

            return tuple(data)


        elif self.phase == 'val' or 'test':
            pic = sorted(os.listdir(join(self.data_dir, self.image_list[index])))[0]
            img = Image.open(join(self.data_dir, self.image_list[index], pic))
            w, h = 496,496

            image = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))), 3, w, h)
            label = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))), w, h)
            imt = torch.zeros(len(os.listdir(join(self.data_dir, self.image_list[index]))), w, h)

            for i, pic_name in enumerate(sorted(os.listdir(join(self.data_dir, self.image_list[index])))):
                data = [Image.open(join(self.data_dir, self.image_list[index], pic_name))]

                imt_3 = torch.from_numpy(np.array(data[0].resize((h,w))).transpose(2,0,1))
                #
                imt_i = imt_3[1]
                imt[i,:,:] = imt_i

                label_name = str(int(pic_name.split('.')[0])) + '.bmp'
                data.append(Image.open(join(self.data_dir, self.label_list[index], label_name)))
                data = list(self.transforms(*data))
                image[i, :, :, :] = data[0]
                label[i, :, :] = data[1]

            return (image, label.long(), imt)


    def __len__(self):
        return len(self.image_list)


    def read_lists(self, device, hist_equal):
        if device == 'all':
            image_path = join(self.list_dir, self.phase + '_images.txt')
            label_path = join(self.list_dir, self.phase + '_labels.txt')
            layer_path = join(self.list_dir, self.phase + '_layers.txt')

        else:
            image_path = join(self.list_dir, self.phase + '_images_'+ device+'.txt')
            label_path = join(self.list_dir, self.phase + '_labels_' + device + '.txt')
            layer_path = join(self.list_dir, self.phase + '_layers_' + device + '.txt')

        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        self.label_list = [line.strip() for line in open(label_path, 'r')]
        if self.phase == 'train':
            self.layer_list = [line.strip() for line in open(layer_path, 'r')]
            assert len(self.image_list) == len(self.label_list) == len(self.layer_list)
        assert len(self.image_list) == len(self.label_list)

        if self.phase == 'train':
            print('Total train image is : %d' % len(self.image_list))
        else:
            print('Total val pid is : %d' % len(self.image_list))
