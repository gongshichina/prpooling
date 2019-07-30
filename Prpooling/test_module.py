# -*- coding: utf-8 -*-
# File   : test_prroi_pooling2d.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 18/02/2018
#
# This file is part of Jacinle.

import torch
import torch.nn as nn
import torch.nn.functional as F

from prroi_pool import PrRoIPool2D


class IoUNet(nn.Module):
    def __init__(self):
        super(IoUNet, self).__init__()
        pool = PrRoIPool2D(7, 7, spatial_scale=0.5)

    def forward():
        out = pool(features, rois)


if __name__ == '__main__':

        pool = PrRoIPool2D(2, 2, spatial_scale=0.5)

        features = torch.rand((4, 2, 24, 32)).cuda()
        rois = torch.tensor([
            [0, 0, 0, 4, 4],
            [1, 14, 14, 18, 18],
        ]).float().cuda()
        features.requires_grad = rois.requires_grad = True

        out = pool(features, rois)
        loss = out.sum()
        loss.backward()
        print(out)
        print(loss)
        print(rois.grad)
        