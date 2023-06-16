from __future__ import division

import torch
import torch.nn as nn
import logging
from scipy.io import loadmat,savemat

from PIL import Image, ImageOps
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torchvision.transforms import Compose
from torchvision import transforms
import random
import numpy as np
import h5py
import cv2


class HyperValid_ICVL(Dataset):
    def __init__(self, args):
        super(HyperValid_ICVL, self).__init__()

        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)
        self.mat_names.sort()

        self.rgb_path = args.rgb_path_valid
        self.rgb_names = os.listdir(self.rgb_path)


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        hsi = mat['HyperImage']
        print(mat_name)

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4] + "_clean.png")
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float64) / 255.0

        hsi = torch.from_numpy(hsi.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()

        _, H, W = rgb.shape
        print(H, W)


        H = (1392 // 4) * 4
        W = (1300 // 4) * 4

        h_crop = 128
        w_crop = 128

        rgb_list = []
        hsi_list = []
        for h_index in range(0, H, h_crop):
            for w_index in range(0, W, w_crop):
                rgb_patch = rgb[:, h_index: h_index + h_crop, w_index: w_index + w_crop]
                hsi_patch = hsi[:, h_index: h_index + h_crop, w_index: w_index + w_crop]
                rgb_list.append(rgb_patch)
                hsi_list.append(hsi_patch)

        return rgb_list, hsi_list


class HyperValid_NTIRE20_clean(Dataset):
    def __init__(self, args):
        super(HyperValid_NTIRE20_clean, self).__init__()

        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path_valid
        self.rgb_names = os.listdir(self.rgb_path)


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        # print(self.rgb_path)
        # print(self.mat_names[index][:-4])
        print(mat_name)
        hsi = mat['cube']


        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4] + "_clean.png")
        # print(rgb_name)

        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float64) / 255.0

        h, w, _ = hsi.shape
        print(h, w)

        hsi_1 = hsi[0:256, 0:256, :]
        rgb_1 = rgb[0:256, 0:256, :]

        hsi_2 = hsi[0:256, 256:(w//4)*4, :]
        rgb_2 = rgb[0:256, 256:(w//4)*4, :]

        hsi_3 = hsi[256:(h // 4) * 4, 0:256, :]
        rgb_3 = rgb[256:(h // 4) * 4, 0:256, :]

        hsi_4 = hsi[256:(h//4)*4, 256:(w//4)*4, :]
        rgb_4 = rgb[256:(h//4)*4, 256:(w//4)*4, :]

        hsi_1 = torch.from_numpy(hsi_1.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb_1 = torch.from_numpy(rgb_1.astype(np.float32).transpose(2, 0, 1)).contiguous()

        hsi_2 = torch.from_numpy(hsi_2.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb_2 = torch.from_numpy(rgb_2.astype(np.float32).transpose(2, 0, 1)).contiguous()

        hsi_3 = torch.from_numpy(hsi_3.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb_3 = torch.from_numpy(rgb_3.astype(np.float32).transpose(2, 0, 1)).contiguous()

        hsi_4 = torch.from_numpy(hsi_4.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb_4 = torch.from_numpy(rgb_4.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return rgb_1, hsi_1, rgb_2, hsi_2, rgb_3, hsi_3, rgb_4, hsi_4


class HyperValid_NTIRE20_real(Dataset):
    def __init__(self, args):
        super(HyperValid_NTIRE20_real, self).__init__()

        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)

        self.rgb_path = args.rgb_path_valid
        self.rgb_names = os.listdir(self.rgb_path)


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        print(mat_name)
        hsi = mat['cube']

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4] + "_RealWorld.jpg")
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float64) / 255.0

        h, w, _ = hsi.shape
        hsi_1 = hsi[0:256, 0:256, :]
        rgb_1 = rgb[0:256, 0:256, :]

        hsi_2 = hsi[0:256, 256:(w // 4) * 4, :]
        rgb_2 = rgb[0:256, 256:(w // 4) * 4, :]

        hsi_3 = hsi[256:(h // 4) * 4, 0:256, :]
        rgb_3 = rgb[256:(h // 4) * 4, 0:256, :]

        hsi_4 = hsi[256:(h // 4) * 4, 256:(w // 4) * 4, :]
        rgb_4 = rgb[256:(h // 4) * 4, 256:(w // 4) * 4, :]

        hsi_1 = torch.from_numpy(hsi_1.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb_1 = torch.from_numpy(rgb_1.astype(np.float32).transpose(2, 0, 1)).contiguous()

        hsi_2 = torch.from_numpy(hsi_2.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb_2 = torch.from_numpy(rgb_2.astype(np.float32).transpose(2, 0, 1)).contiguous()

        hsi_3 = torch.from_numpy(hsi_3.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb_3 = torch.from_numpy(rgb_3.astype(np.float32).transpose(2, 0, 1)).contiguous()

        hsi_4 = torch.from_numpy(hsi_4.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb_4 = torch.from_numpy(rgb_4.astype(np.float32).transpose(2, 0, 1)).contiguous()

        return rgb_1, hsi_1, rgb_2, hsi_2, rgb_3, hsi_3, rgb_4, hsi_4


class HyperTest_NTIRE18_clean(Dataset):
    def __init__(self, args):
        super(HyperTest_NTIRE18_clean, self).__init__()
        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)
        self.mat_names.sort()

        self.rgb_path = args.rgb_path_valid
        self.rgb_names = os.listdir(self.rgb_path)


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        hsi = mat['rad']
        hsi = np.array(hsi)
        hsi = hsi.transpose(2, 1, 0)
        hsi = hsi / 4095.0

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4] + "_clean.png")
        rgb = cv2.imread(rgb_name)
        rgb = rgb.astype(np.float64) / 255.0

        hsi = torch.from_numpy(hsi.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()

        _, H, W = rgb.shape
        print(H,W)

        h_crop = 256
        w_crop = 256
        # h_crop = 100
        # w_crop = 100

        # h_crop = 300
        # w_crop = 300


        rgb = rgb[:, : (H//16)*16, : (W//16)*16]
        hsi = hsi[:, : (H//16)*16, : (W//16)*16]

        rgb_list = []
        hsi_list = []
        for h_index in range(0, H, h_crop):
            for w_index in range(0, W, w_crop):
                rgb_patch = rgb[:, h_index: h_index + h_crop, w_index: w_index + w_crop]
                hsi_patch = hsi[:, h_index: h_index + h_crop, w_index: w_index + w_crop]
                rgb_list.append(rgb_patch)
                hsi_list.append(hsi_patch)

        return rgb_list, hsi_list


class HyperTest_NTIRE18_real(Dataset):
    def __init__(self, args):
        super(HyperTest_NTIRE18_real, self).__init__()

        self.mat_path = args.mat_path_valid
        self.mat_names = os.listdir(self.mat_path)
        self.mat_names.sort()

        self.rgb_path = args.rgb_path_valid
        self.rgb_names = os.listdir(self.rgb_path)


    def __len__(self):
        return len(self.mat_names)

    def __getitem__(self, index):
        mat_name = os.path.join(self.mat_path, self.mat_names[index])
        mat = loadmat(mat_name)
        hsi = mat['rad']
        hsi = np.array(hsi)
        hsi = hsi.transpose(2, 1, 0)
        hsi = hsi / 4095.0

        # (1300, 1392, 31), in range [0, 1], float32

        rgb_name = os.path.join(self.rgb_path, self.mat_names[index][:-4] + "_camera.jpg")
        rgb = cv2.imread(rgb_name).astype(np.float64)
        rgb = rgb / 255.0

        hsi = torch.from_numpy(hsi.astype(np.float32).transpose(2, 0, 1)).contiguous()
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)).contiguous()

        _, H, W = rgb.shape

        rgb = rgb[:, : (H//16)*16, : (W//16)*16]
        hsi = hsi[:, : (H//16)*16, : (W//16)*16]

        # h_crop = 128
        # w_crop = 128

        # h_crop = 300
        # w_crop = 300

        h_crop = 256
        w_crop = 256

        rgb_list = []
        hsi_list = []
        for h_index in range(0, H, h_crop):
            for w_index in range(0, W, w_crop):
                rgb_patch = rgb[:, h_index: h_index + h_crop, w_index: w_index + w_crop]
                hsi_patch = hsi[:, h_index: h_index + h_crop, w_index: w_index + w_crop]
                rgb_list.append(rgb_patch)
                hsi_list.append(hsi_patch)

        return rgb_list, hsi_list

