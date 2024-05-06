#!/usr/bin/env python



import os
import sys
import copy
import time
import datetime
from math import exp

import numpy as np
import random

from torch.autograd import Variable
from tqdm import tqdm, trange
from collections import OrderedDict
import SimpleITK as sitk

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary
# import graphviz
# import hiddenlayer as h


# from lib.loss import *
import lib.utils as utils
# import lib.transforms as med_transform
# import lib.datasets as med_data
from lib.network_factory import get_network
import lib.visualize as vis
import lib.evalMetrics as metrics
from lib.param_dict import save_dict_to_json, load_jason_to_dict

# # for distribute training
# import torch.distributed as dist
# import torch.nn as nn

sys.path.append(os.path.realpath(".."))
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


class BaseExperiment_():
    def __init__(self):
        pass

    def _ssim_3D(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return 1 - ret
    def create_window_3D(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                      window_size).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()