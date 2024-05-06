#!/usr/bin/env python
"""
last modified by oylei98@163.com on 2021/12/20
"""

import os
import sys
import copy
import time
import datetime

import numpy as np
import random
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


from lib.loss import *
import lib.utils as utils
import lib.transforms as med_transform
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


class BaseExperiment():
    def __init__(self, config, **kwargs):
        self.config = config
        pass

    def setup_log(self):
        pass

    def setup_random_seed(self):
        torch.manual_seed(self.config['random_seed'])
        torch.cuda.manual_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        random.seed(self.config['random_seed'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def setup_train_data(self):
        pass

    def setup_model(self):
        pass

    # def setup_loss(self):
    #     pass

    # def setup_optimizer(self):
    #     pass

    def setup_train(self):
        self.setup_log()
        self.setup_random_seed()
        self.setup_model()
        self.setup_train_data()

    def train(self, **kwargs):
        raise NotImplementedError()

    def train_one_epoch(self, **kwargs):
        raise NotImplementedError()

    def validate(self, **kwargs):
        raise NotImplementedError()
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
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

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


    @staticmethod
    def save_checkpoint(state, is_best, path, prefix=None, name='checkpoint.pth', max_keep=1):
        if not os.path.exists(path):
            os.makedirs(path)
        name = '_'.join([prefix, name]) if prefix else name
        best_name = '_'.join([prefix, 'model_best.pth']) if prefix else 'model_best.pth'
        torch.save(state, os.path.join(path, name))
        if is_best:
            torch.save(state, os.path.join(path, best_name))

    @staticmethod
    def initialize_model(model, optimizer=None, ckpoint_path=None):
        """
        Initilaize a reg_model with saved checkpoins, or random values
        :param model: a pytorch reg_model to be initialized
        :param optimizer: optional, optimizer whose parameters can be restored from saved checkpoints
        :param ckpoint_path: The path of saved checkpoint
        :return: currect epoch and best validation score
        """
        finished_epoch = 0
        best_score = 0

        # initialize model by existed parameters
        if ckpoint_path:
            if os.path.isfile(ckpoint_path):
                print("=> loading checkpoint '{}'".format(ckpoint_path))
                checkpoint = torch.load(ckpoint_path, map_location=next(
                    model.parameters()).device)
                print (checkpoint.keys())
                if 'best_score' in checkpoint:
                    best_score = checkpoint['best_score']
                elif 'reg_best_score' in checkpoint:
                    best_score = checkpoint['reg_best_score']
                elif 'seg_best_score' in checkpoint:
                    best_score = checkpoint['seg_best_score']
                else:
                    raise ValueError('no best score key')

                if type(best_score) is torch.Tensor:
                    best_score = best_score.cpu().item()

                # processing state dict because of multi gpu
                state_dict = checkpoint['model_state_dict']
                new_state_dcit = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dcit[k.replace('module.', '')] = v

                model.load_state_dict(new_state_dcit, strict=True)
                model = model.cuda()
                # if optimizer and checkpoint.__contains__('optimizer_state_dict'):
                #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                finished_epoch = finished_epoch + checkpoint['epoch']
                print("=> loaded checkpoint '{}' (epoch {})".format(ckpoint_path, checkpoint['epoch']))
                del checkpoint
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(ckpoint_path))

        # initialize model by random parameters
        else:
            # reg_model.apply(weights_init)
            model.weights_init()

        return finished_epoch, best_score

    @staticmethod
    def save_checkpoint(state, is_best, path, prefix=None, name='checkpoint.pth', max_keep=1):
        if not os.path.exists(path):
            os.makedirs(path)
        name = '_'.join([prefix, name]) if prefix else name
        best_name = '_'.join([prefix, 'model_best.pth']) if prefix else 'model_best.pth'
        torch.save(state, os.path.join(path, name))
        if is_best:
            torch.save(state, os.path.join(path, best_name))


    @staticmethod
    def initialize_model_(model, optimizer=None, ckpoint_path=None):
        """
        Initilaize a reg_model with saved checkpoins, or random values
        :param model: a pytorch reg_model to be initialized
        :param optimizer: optional, optimizer whose parameters can be restored from saved checkpoints
        :param ckpoint_path: The path of saved checkpoint
        :return: currect epoch and best validation score
        """

        # initialize model by existed parameters
        if ckpoint_path:
            if os.path.isfile(ckpoint_path):
                # print("=> loading checkpoint '{}'".format(ckpoint_path))
                # checkpoint = model.load_state_dict(torch.load(ckpoint_path, map_location=next(
                #     model.parameters()).cuda()))
                # print(checkpoint.keys())
                # # processing state dict because of multi gpu
                # state_dict = checkpoint['model_state_dict']
                # new_state_dcit = OrderedDict()
                # for k, v in state_dict.items():
                #     new_state_dcit[k.replace('module.', '')] = v
                #
                # model.load_state_dict(new_state_dcit, strict=True)
                model.load_state_dict(torch.load(ckpoint_path, map_location=next(
                    model.parameters()).device))
                model = model.cuda()
                # if optimizer and checkpoint.__contains__('optimizer_state_dict'):
                #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # print("=> loaded checkpoint '{}' (epoch {})".format(ckpoint_path, checkpoint['epoch']))
                # print("=> loaded checkpoint '{}' (epoch)".format(ckpoint_path))
                # del checkpoint
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(ckpoint_path))

        # initialize model by random parameters
        else:
            # reg_model.apply(weights_init)
            model.weights_init()

