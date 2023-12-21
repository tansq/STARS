# Color dataset for binary classification
# Author: QIUSHI LI
# TIME: 2021-6-8

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import scipy.io as sio
from PIL import Image
import random
import cv2
import copy

from .color_prepoc import decode_from_dct


# class ToTensor():
#     def __call__(self, data):
#         # data = data.astype(np.float32)
#         # data = np.expand_dims(data, axis=1)   # for grayscale
#         return torch.from_numpy(data)


class CustomDataset_color(Dataset):
    def __init__(self, filename, cover_dir='', stego_dir='',
                 transform=None, train_flag=False, img_type=None):

        self.image_label_list = self.read_file(filename, cover_dir, stego_dir)
        # self.cover_dir = cover_dir
        # self.stego_dir = stego_dir
        self.len = len(self.image_label_list)

        self.toTensor = transforms.ToTensor()
        self.transform = transform
        self.rot = 0
        self.rand_flip = 0
        self.train_flag = train_flag
        self.img_type = img_type


    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_path, label = self.image_label_list[index]

        label = np.array(label)

        if self.train_flag:
            if index % 2 == 0:
                self.rot = random.randint(0, 3)
                self.rand_flip = random.random()
                img = self.load_data(image_path)
            else:
                img = self.load_data(image_path)
        else:
            img = self.load_data(image_path)

        return img, label

    def __len__(self):
        data_len = len(self.image_label_list)
        return data_len

    def read_file(self, filename, cover_dir, stego_dir):

        with open(filename) as f:
            lines = f.readlines()
            lines = [a.strip() for a in lines]
            if lines[0].startswith('/'):
                cover_list = copy.deepcopy(lines)
                stego_list = [a.replace(cover_dir, stego_dir) for a in lines]
                if 'color_spatial' in stego_dir:
                    stego_list = [a.replace('.tif', '.ppm') for a in stego_list]
            else:
                lines = [os.path.basename(a.strip()).split('.')[0] for a in lines]
                if 'color_noround_ycrcb' in stego_dir:
                    cover_ext = '.npy'
                    stego_ext = '.npy'
                elif 'color_spatial' in stego_dir:
                    cover_ext = '.tif'
                    stego_ext = '.ppm'
                elif 'npz' in stego_dir:
                    cover_ext = '.npz'
                    stego_ext = '.npz'
                else:
                    cover_ext = '.ppm'
                    stego_ext = '.ppm'
                stego_list = [os.path.join(stego_dir, a + stego_ext) for a in lines]
                cover_list = [os.path.join(cover_dir, a + cover_ext) for a in lines]

        image_label_list = []

        for cover_file, stego_file in zip(cover_list, stego_list):
            image_label_list.append([cover_file, 0])
            image_label_list.append([stego_file, 1])
        return image_label_list

    def load_data(self, path):

        assert os.path.exists(path), "%s dosen't exist!"%(path)
        if path.endswith('.mat'):
            np_img = sio.loadmat(path)['im']
            img = Image.fromarray(np_img)
            image = self.toTensor(img)
        elif path.endswith('.npy'):
            np_img = np.load(path).astype(np.float32)
            if self.img_type == 'rgb':
                np_img = cv2.cvtColor(np_img, cv2.COLOR_YCrCb2RGB)
            # img = Image.fromarray(np_img)
            image = np.transpose(np_img, (2, 0, 1))
            image = torch.from_numpy(image)
        elif path.endswith('npz'):
            image = decode_from_dct(path, img_comp='ycrcb')
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        else:
            img = Image.open(path)
            image = self.toTensor(img)
        # if self.transform is not None:
        #     img = self.transform(img)
        # image = self.toTensor(img)

        if self.train_flag:
            if self.rand_flip < 0.5:
                image = torch.rot90(image, self.rot, dims=[1, 2])
            else:
                image = torch.flip(torch.rot90(image, self.rot, dims=[1, 2]), dims=[2])  # horizontal flip

        return image

    def shuffle_pair(self, nopc=False):
        cover_list = self.image_label_list[::2]
        stego_list = self.image_label_list[1::2]

        if nopc:  # no pair constraint
            random.shuffle(cover_list)
            random.shuffle(stego_list)
        else:
            tmp_list = list(zip(cover_list, stego_list))
            random.shuffle(tmp_list)
            cover_list, stego_list = list(zip(*tmp_list))

        self.image_label_list = []
        for i in range(len(cover_list)):
            self.image_label_list.append(cover_list[i])
            self.image_label_list.append(stego_list[i])
