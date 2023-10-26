#!/usr/bin/env python
"""
Created by zhenlinx on 11/10/19
"""
# import torch
# import numpy as np
import os
import sys
# import shutil
# import time
# import datetime
# import gc
# import subprocess
import sklearn.model_selection as model_selection
import random
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.data import DataLoader
# from torchvision import transforms
# # import torchvision.vision_utils as vision_utils
# from torch.autograd import Variable
# import torch.nn.functional as F
#
# from tensorboardX import SummaryWriter
# import SimpleITK as sitk

sys.path.append(os.path.realpath(".."))
# import utils.loss as bio_loss
# from utils.loss import get_loss_function
# import utils.evalMetrics as metrics
# import utils.visualize as vis
# from misc.module_parameters import save_dict_to_json


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
    if ckpoint_path:
        if os.path.isfile(ckpoint_path):
            print("=> loading checkpoint '{}'".format(ckpoint_path))
            checkpoint = torch.load(ckpoint_path, map_location=next(
                model.parameters()).device)
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

            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            if optimizer and checkpoint.__contains__('optimizer_state_dict'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            finished_epoch = finished_epoch + checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpoint_path, checkpoint['epoch']))
            del checkpoint
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(ckpoint_path))
    else:
        # reg_model.apply(weights_init)
        model.weights_init()
    return finished_epoch, best_score


def get_identity_transform_batch(size, normalize=True):
    """
    generate an identity transform for given image size (NxCxDxHxW)
    :param size: Batch, D,H,W size
    :param normalize: normalized index into [-1,1]
    :return: identity transform with size Nx3xDxHxW
    """
    _identity = get_identity_transform(size[2:], normalize)
    return _identity


def get_identity_transform(size, normalize=True):
    """

    :param size: D,H,W size
    :param normalize:
    :return: 3XDxHxW tensor
    """
    gg = size
    if normalize:
        xx, yy, zz = torch.meshgrid([torch.arange(0, size[k]).float() / (size[k] - 1) * 2.0 - 1 for k in [0, 1, 2]])
    else:
        xx, yy, zz = torch.meshgrid([torch.arange(0, size[k]) for k in [0, 1, 2]])
    _identity = torch.stack([zz, yy, xx])
    return _identity


def dataset():
    """
    create data list in txt format
    """
    data_root = '/home/qulab/disk/OYL/data/mindboggle/'
    print('error dir' if not os.path.isdir(data_root) else 'os.walk')
    training_list = []
    for root, sub, file in os.walk(data_root):
        if len(sub) > 5:  # inner dir
            for inner in sub:
                if root.endswith('Extra-18_volumes'):
                    if inner.lower().startswith('after') or inner.lower().startswith('hln-12-3') \
                            or inner.lower().startswith('twins'):
                        training_list.append(os.path.join(root, inner))
                elif root.endswith('MMRR-21_volumes'):
                    if not inner.endswith('21-21') and not inner.endswith('21-1'):
                        training_list.append(os.path.join(root, inner))
                elif root.endswith('NKI-RS-22_volumes'):
                    if not inner.endswith('22-16'):
                        training_list.append(os.path.join(root, inner))
                elif root.endswith('NKI-TRT-20_volumes'):
                    training_list.append(os.path.join(root, inner))
                elif root.endswith('OASIS-TRT-20_volumes'):
                    training_list.append(os.path.join(root, inner))
    random.shuffle(training_list)
    training_list.remove('/home/qulab/disk/OYL/data/mindboggle/NKI-RS-22_volumes/NKI-RS-22-10')
    print(len(training_list))
    train, test = model_selection.train_test_split(training_list, train_size=0.76, shuffle=True)
    train.insert(0, '/home/qulab/disk/OYL/data/mindboggle/NKI-RS-22_volumes/NKI-RS-22-10')
    test, val = model_selection.train_test_split(test, test_size=0.25, shuffle=True)
    with open(data_root + 'train.txt', 'w') as fp:
        for i in train:
            fp.write(i + '\n')
    with open(data_root + 'test.txt', 'w') as fp:
        for i in test:
            fp.write(i + '\n')
    with open(data_root + 'val.txt', 'w') as fp:
        for i in val:
            fp.write(i + '\n')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)