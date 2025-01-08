from .transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import geoseg.datasets.blu_transform as transform
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import random
from skimage import io
num_classes = 7
BLU_COLORMAP = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [125, 125, 125], [0, 0, 0]]
BLU_CLASSES = ['Background', 'Building', 'Vegetation', 'Water', 'Farmland', 'Road', 'Invalid']
BLU_MEAN = np.array([122.19, 121.35, 117.29])
BLU_STD = np.array([63.26, 62.45, 61.65])

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)


def get_training_transform():
    train_transform = [
        # albu.Resize(height=1024, width=1024),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        # albu.RandomRotate90(p=0.5),
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        #     albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
        # ], p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # multi-scale training and crop
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)])
    img, mask = crop_aug(img, mask)

    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def normalize_image(im):
    return (im - BLU_MEAN) / BLU_STD


def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(BLU_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Index2Color(pred):
    colormap = np.asarray(BLU_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def Colorls2Index(ColorLabels):
    for i, data in enumerate(ColorLabels):
        ColorLabels[i] = Color2Index(data)
    return ColorLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap <= num_classes)
    return IndexMap.astype(np.uint8)


def read_RSimages(data_dir, mode, rescale_ratio=False):
    print('Reading data from ' + data_dir + ':')
    assert mode in ['train', 'val', 'test']
    data_list = []
    img_dir = os.path.join(data_dir, mode, 'image')
    item_list = os.listdir(img_dir)
    for item in item_list:
        if (item[-4:] == '.png'): data_list.append(os.path.join(img_dir, item))
    data_length = int(len(data_list))
    count = 0
    data, labels = [], []
    for it in data_list:
        # print(it)
        img_path = it
        mask_path = img_path.replace('image', 'label')
        img = io.imread(img_path)
        label = Color2Index(io.imread(mask_path))
        if rescale_ratio:
            img = rescale_image(img, rescale_ratio, 2)
            label = rescale_image(label, rescale_ratio, 0)
        data.append(img)
        labels.append(label)
        count += 1
        if not count % 10: print('%d/%d images loaded.' % (count, data_length))
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    return data, labels


def rescale_images(imgs, scale, order=0):
    rescaled_imgs = []
    for im in imgs:
        rescaled_imgs.append(rescale_image(im, scale, order))
    return rescaled_imgs


def rescale_image(img, scale=8, order=0):
    flag = cv2.INTER_NEAREST
    if order == 1:
        flag = cv2.INTER_LINEAR
    elif order == 2:
        flag = cv2.INTER_AREA
    elif order > 2:
        flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0] / scale), int(img.shape[1] / scale)), interpolation=flag)
    return im_rescaled

class bluDataset(Dataset):
    def __init__(self,
                 data_dir, mode, random_crop=False, crop_nums=40, random_flip=False, sliding_crop=False,
                 size_context=256 * 3, size_local=256, scale=4):
        self.size_context = size_context
        self.size_local = size_local
        self.scale = scale
        self.crop_nums = crop_nums
        self.random_flip = random_flip
        self.random_crop = random_crop
        data, labels = read_RSimages(data_dir, mode)
        self.data = data
        self.labels = labels

        data_s = rescale_images(data, scale, 2)
        labels_s = rescale_images(labels, scale, 0)
        padding_size = (size_context - size_local) / scale / 2
        self.data_s, self.labels_s = transform.data_padding_fixsize(data_s, labels_s, [padding_size, padding_size])
        if sliding_crop:
            self.data_s, self.labels_s, self.data, self.labels = transform.slidding_crop_WC(self.data_s, self.labels_s,
                                                                                            self.data, self.labels,
                                                                                            size_context, size_local,
                                                                                            scale)

        if self.random_crop:
            self.len = crop_nums * len(self.data)
        else:
            self.len = len(self.data)

    def __getitem__(self, idx):
        if self.random_crop:
            idx = int(idx / self.crop_nums)
            data_s, label_s, data, label = transform.random_crop2(self.data_s[idx], self.labels_s[idx],
                                                                  self.data[idx], self.labels[idx], self.size_context,
                                                                  self.size_local, self.scale)
        else:
            data = self.data[idx]
            label = self.labels[idx]
            data_s = self.data_s[idx]
            label_s = self.labels_s[idx]
        if self.random_flip:
            data_s, label_s, data, label = transform.rand_flip2(data_s, label_s, data, label)

        data_s = normalize_image(data_s)
        data_s = torch.from_numpy(data_s.transpose((2, 0, 1)))
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data_s, label_s, data, label

    def __len__(self):
        return self.len