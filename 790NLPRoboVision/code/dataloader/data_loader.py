from __future__ import print_function
import copy

import torch.utils.data as data
import random
from PIL import Image
from dataloader import preprocess
from dataloader import readpfm as rp
import numpy as np
import math

# train/ validation image crop size constants
# DEFAULT_TRAIN_IMAGE_HEIGHT = 256
# DEFAULT_TRAIN_IMAGE_WIDTH = 512
DEFAULT_TRAIN_IMAGE_HEIGHT = 540
DEFAULT_TRAIN_IMAGE_WIDTH = 960


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class SceneflowLoader(data.Dataset):
    def __init__(self, left_images, right_images, left_disparity, left_cam, right_cam, network_downsample_scale, training, loader=default_loader, dploader=disparity_loader):

        self.left_images = left_images
        self.right_images = right_images
        self.left_disparity = left_disparity
        self.left_cam = left_cam
        self.right_cam = right_cam
        self.loader = loader
        self.dploader = dploader
        self.training = training

        # network_downsample_scale denotes maximum times the image features are downsampled by the network.
        # Since the image size used for evaluation may not be divisible by the network_downsample_scale,
        # we pad it with zeros, so that it becomes divible and later unpad the extra zeros.
        self.downsample_scale = network_downsample_scale

    def __getitem__(self, index):
        left_img_fn = self.left_images[index]
        right_img_fn = self.right_images[index]
        left_disp = self.left_disparity[index]
        left_cam = copy.deepcopy(self.left_cam[index])
        right_cam = copy.deepcopy(self.right_cam[index])

        left_img = self.loader(left_img_fn)
        right_img = self.loader(right_img_fn)
        left_disp, left_scale = self.dploader(left_disp)
        left_disp = np.ascontiguousarray(left_disp, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = DEFAULT_TRAIN_IMAGE_HEIGHT, DEFAULT_TRAIN_IMAGE_WIDTH

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            left_disp = left_disp[y1:y1 + th, x1:x1 + tw]
            left_cam['intrinsics']['cx'] -= x1
            left_cam['intrinsics']['cy'] -= y1
            right_cam['intrinsics']['cx'] -= x1
            right_cam['intrinsics']['cy'] -= y1

            processed = preprocess.get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, left_disp, left_cam, right_cam, left_img_fn, right_img_fn
        else:
            w, h = left_img.size

            dw = w + (self.downsample_scale - (w%self.downsample_scale + (w%self.downsample_scale==0)*self.downsample_scale))
            dh = h + (self.downsample_scale - (h%self.downsample_scale + (h%self.downsample_scale==0)*self.downsample_scale))

            # if w-dw < 0, crop() will pad with black pixels
            left_img = left_img.crop((w - dw, h - dh, w, h))
            right_img = right_img.crop((w - dw, h - dh, w, h))
            left_disp_tmp = np.zeros((max(dh, h), max(dw, w)), dtype=np.float32)
            sh = max(0, dh - h) // 2
            sw = max(0, dw - w) // 2
            left_disp_tmp[sh:(sh+h), sw:(sw+w)] = left_disp
            sh = max(0, h - dh) // 2
            sw = max(0, w - dw) // 2
            left_disp = left_disp_tmp[sh:(sh+dh), sw:(sw+dw)]
            left_cam['intrinsics']['cx'] += dw - w
            left_cam['intrinsics']['cy'] += dh - h
            right_cam['intrinsics']['cx'] += dw - w
            right_cam['intrinsics']['cy'] += dh - h

            processed = preprocess.get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, left_disp, dw-w, dh-h, left_cam, right_cam, left_img_fn, right_img_fn

    def __len__(self):
        return len(self.left_images)