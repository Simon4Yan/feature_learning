# encoding: utf-8
"""
@author:  weijian
@contact: dengwj16@gmail.com
"""

import torchvision.transforms as T

import torch

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING), # pad=10
            T.RandomCrop([256, 256]), #  scale=(0.5, 1.0) default 0.05 to 1
            #T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),# light
            T.ToTensor(),
            normalize_transform
        ])

    else:
        #cfg.INPUT.SIZE_TEST
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
