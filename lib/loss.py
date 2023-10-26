import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

# from models.base import BaseExperiment
from .transforms import mask_to_one_hot
from .utils_ import *
from math import exp

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]
class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()

    def forward(self, predict, gt):
        predict = predict.squeeze().flatten().unsqueeze(dim=0)
        out = torch.mean()
        return out


class DiceLoss(Function):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.000001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.siz-e())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
        #       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2 * eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
        #     union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2 * IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect / (union * union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input, None


def dice_loss(input, target):
    return DiceLoss()(input, target)


def dice_error(input, target):
    eps = 0.000001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2 * eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
    #    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
    #        union, intersect, target_sum, result_sum, 2*IoU))
    return 2 * IoU


class SoftCrossEntropy(nn.Module):
    """Cross Entropy that allows target to be probabilistic input cross classes"""

    def __init__(self, n_class=None, weight_type='Simple', no_bg=False, softmax=False):
        super(SoftCrossEntropy, self).__init__()
        self.weight_type = weight_type
        self.n_class = n_class
        self.no_bg = no_bg
        self.softmax = softmax  # if the source inputs are in 0~1 range

    def forward(self, pred, target):
        """

        :param pred: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param target: Tensor, ground truth mask when of size BxDxMxN;
                             or class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        shape = list(pred.shape)

        # flat the spatial dimensions
        source_flat = pred.view(shape[0], shape[1], -1)

        # flat the spatial dimensions and transform it into one-hot coding
        if len(target.shape) == len(shape) - 1:
            target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
        elif target.shape[1] == shape[1]:
            target_flat = target.view(shape[0], shape[1], -1)
        else:
            target_flat = None
            raise ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape,
                                                                                        shape[:1] + [1, ] + shape[2:]))

        if self.softmax:
            return torch.mean(torch.sum(- target * F.log_softmax(pred, 1), 1))
        else:
            return torch.mean(torch.sum(- target * torch.log(pred.clamp_(min=1e-8)), 1))


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, soft_max=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.soft_max = soft_max

    def forward(self, inputs, targets):
        """

        :param inputs: Bxn_classxXxYxZ
        :param targets: Bx.....  , range(0,n_class)
        :return:
        """
        # squeeze the targets from Bx1x...to Bx...
        # if len(inputs.shape) == len(targets.shape) and targets.shape[1] == 1:
        #     targets = targets.squeeze(1)


        if len(inputs.shape) > 2 and len(targets.shape) > 1:
            inputs = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, inputs.size(1))
            targets = targets.view(-1)

        targets =targets.long()

        # TODO avoid softmax in focal loss
        if self.soft_max:
            P = F.softmax(inputs, dim=1)
        else:
            P = inputs


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[targets.data.view(-1)].view(-1)

        log_p = - F.cross_entropy(inputs, targets, reduce=False)
        probs = F.nll_loss(P, targets, reduce=False)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class SoftFocalLoss(FocalLoss):
    """
    Focal Loss that takes probabilistic map targets
    """
    def forward(self, inputs, targets):
        pass


class FocalLoss2(nn.Module):
    """
    https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
    """

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss

        return loss.sum()


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)


"""
segmentation loss
"""


# class DiceLoss(nn.Module):
#     def initialize(self, class_num, weight=None):
#         self.class_num = class_num
#         self.class_num = class_num
#         if weight is None:
#             self.weight = torch.ones(class_num, 1) / self.class_num
#         else:
#             self.weight = weight
#         self.weight = torch.squeeze(self.weight)
#
#     def forward(self, input, target, inst_weights=None, train=None):
#         """
#         input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
#         target is a Bx....   range 0,1....N_label
#         """
#         in_sz = input.size()
#         from functools import reduce
#         extra_dim = reduce(lambda x, y: x * y, in_sz[2:])
#         targ_one_hot = torch.zeros(in_sz[0], in_sz[1], extra_dim).cuda()
#         targ_one_hot.scatter_(1, target.view(in_sz[0], 1, extra_dim), 1.)
#         target = targ_one_hot.view(in_sz).contiguous()
#         probs = F.softmax(input, dim=1)
#         num = probs * target
#         num = num.view(num.shape[0], num.shape[1], -1)
#         num = torch.sum(num, dim=2)
#
#         den1 = probs  # *probs
#         den1 = den1.view(den1.shape[0], den1.shape[1], -1)
#         den1 = torch.sum(den1, dim=2)
#
#         den2 = target  # *target
#         den2 = den1.view(den2.shape[0], den2.shape[1], -1)
#         den2 = torch.sum(den2, dim=2)
#         # print("den1:{}".format(sum(sum(den1))))
#         # print("den2:{}".format(sum(sum(den2/den1))))
#
#         dice = 2 * (num / (den1 + den2))
#         dice = self.weight.expand_as(dice) * dice
#         dice_eso = dice
#         # dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
#         dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
#         return dice_total


# class GeneralizedDiceLoss(nn.Module):
#     def initialize(self, class_num, weight=None):
#         self.class_num = class_num
#         if weight is None:
#             self.weight = torch.ones(class_num, 1)
#         else:
#             self.weight = weight
#
#         self.weight = torch.squeeze(self.weight)
#
#     def forward(self, input, target, inst_weights=None, train=None):
#         """
#         input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
#         target is a Bx....   range 0,1....N_label
#         """
#         in_sz = input.size()
#         from functools import reduce
#         extra_dim = reduce(lambda x, y: x * y, in_sz[2:])
#         targ_one_hot = torch.zeros(in_sz[0], in_sz[1], extra_dim).cuda()
#         targ_one_hot.scatter_(1, target.view(in_sz[0], 1, extra_dim), 1.)
#         target = targ_one_hot.view(in_sz).contiguous()
#         probs = F.softmax(input, dim=1)
#         num = probs * target
#         num = num.view(num.shape[0], num.shape[1], -1)
#         num = torch.sum(num, dim=2)  # batch x ch
#
#         den1 = probs
#         den1 = den1.view(den1.shape[0], den1.shape[1], -1)
#         den1 = torch.sum(den1, dim=2)  # batch x ch
#
#         den2 = target
#         den2 = den1.view(den2.shape[0], den2.shape[1], -1)
#         den2 = torch.sum(den2, dim=2)  # batch x ch
#         # print("den1:{}".format(sum(sum(den1))))
#         # print("den2:{}".format(sum(sum(den2/den1))))
#         weights = self.weight.expand_as(den1)
#
#         dice = 2 * (torch.sum(weights * num, dim=1) / torch.sum(weights * (den1 + den2), dim=1))
#         dice_eso = dice
#         # dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg
#         dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
#
#         return dice_total


class DiceLossOnLabel(nn.Module):
    """Dice loss from two inputs of segmentation masks(different with between a mask and a probability map)"""

    def __init__(self, n_class=None, eps=10e-6):
        super(DiceLossOnLabel, self).__init__()
        self.n_class = n_class
        self.eps = eps
        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target, weight_type='Uniform', average=True):
        """
        :param source: Tensor of size Bx1xDxMxN
        :param target: Tensor of size Bx1xDxMxN
        :return:
        """
        assert source.shape == target.shape

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        mask_shape = list(target.shape)
        source_one_hot = mask_to_one_hot(source.view(mask_shape[0], mask_shape[1], -1), self.n_class)
        target_one_hot = mask_to_one_hot(target.view(mask_shape[0], mask_shape[1], -1), self.n_class)

        # does not consider background
        source_one_hot = source_one_hot[:, 1:, :]
        target_one_hot = target_one_hot[:, 1:, :]

        #
        source_volume = source_one_hot.sum(2)
        target_volume = target_one_hot.sum(2)

        if weight_type == 'Simple':
            weights = target_volume.float().reciprocal()
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
        elif weight_type == 'Uniform':
            weights = torch.ones(mask_shape[0], mask_shape[1])

        intersection = source_one_hot * target_one_hot
        scores = (2. * intersection.sum(2).float() * weights) / (
                weights * (source_volume.float() + target_volume.float()) + self.eps)

        return 1 - scores.mean()


class MSELossMultiClass(nn.Module):
    """Dice loss from two inputs of segmentation (between a mask and a probability map)"""

    def __init__(self, n_class=None, weight_type='Uniform', no_bg=False, softmax=False, eps=1e-7):
        super(MSELossMultiClass, self).__init__()
        self.weight_type = weight_type
        self.n_class = n_class
        self.eps = eps
        self.no_bg = no_bg
        self.softmax = softmax  # if the source inputs are in 0~1 range
        # print(self.weight_type, self.n_class, self.eps, self.no_bg, self.softmax)
        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target):
        """

        :param source: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param target: Tensor, ground truth mask when of size BxDxMxN;
                             or class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[-3:] == target.squeeze().shape[-3:]

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        shape = list(source.shape)

        if self.softmax:
            source = F.softmax(source, dim=1)

        # flat the spatial dimensions
        source_flat = source.view(shape[0], shape[1], -1)

        # flat the spatial dimensions and transform it into one-hot coding
        if len(target.shape) == len(shape)-1:
            target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
        elif target.shape[1] == shape[1]:
            target_flat = target.view(shape[0], shape[1], -1)
        else:
            target_flat = None
            raise ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape,
                                                                                        shape[:1] + [1, ] + shape[2:]))
        # does not consider background
        if self.no_bg:
            source_flat = source_flat[:, 1:, :]
            target_flat = target_flat[:, 1:, :]
        # source_volume = source_flat.sum(2)
        # target_volume = target_flat.sum(2)
        #
        if self.weight_type == 'Uniform':
            weights = torch.ones(shape[0], shape[1] - int(self.no_bg))
        else:
            raise ValueError("Class weighting type {} does not exists!".format(self.weight_type))
        # print(weights)
        weights = weights.to(source.device)
        weights = weights / weights.max()
        mse_total_channel = (source_flat - target_flat) ** 2
        mse_mean = (mse_total_channel + self.eps).mean(2)
        mse_mean_ = (weights * mse_mean).sum() / weights.sum()


        return mse_mean_



class DiceLossMultiClass(nn.Module):
    """Dice loss from two inputs of segmentation (between a mask and a probability map)"""

    def __init__(self, n_class=None, weight_type='Simple', no_bg=False, softmax=False, eps=1e-7):
        super(DiceLossMultiClass, self).__init__()
        self.weight_type = weight_type
        self.n_class = n_class
        self.eps = eps
        self.no_bg = no_bg
        self.softmax = softmax  # if the source inputs are in 0~1 range
        # print(self.weight_type, self.n_class, self.eps, self.no_bg, self.softmax)
        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()

    def forward(self, source, target):
        """

        :param source: Tensor of size BxCxDxMxN, One-hot encoding mask /class-wise probability of prediction
        :param target: Tensor, ground truth mask when of size BxDxMxN;
                             or class-wise probability of prediction when of size BxCxDxMxN
        :return:
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[-3:] == target.squeeze().shape[-3:]

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        shape = list(source.shape)

        if self.softmax:
            source = F.softmax(source, dim=1)

        # flat the spatial dimensions
        source_flat = source.view(shape[0], shape[1], -1)
        # flat the spatial dimensions and transform it into one-hot coding
        if len(target.shape) == len(shape)-1:
            target_flat = mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
        elif target.shape[1] == shape[1]:
            target_flat = target.view(shape[0], shape[1], -1)
        else:
            target_flat = None
            raise ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape,
                                                                                        shape[:1] + [1, ] + shape[2:]))

        # does not consider background
        if self.no_bg:
            source_flat = source_flat[:, 1:, :]
            target_flat = target_flat[:, 1:, :]
        #
        source_volume = source_flat.sum(2)
        target_volume = target_flat.sum(2)


        if self.weight_type == 'Simple':
            # weights = (target_volume.float().sqrt() + self.eps).reciprocal()
            weights = (target_volume.float()**(1./3.) + self.eps).reciprocal()
            # temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            # max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            # weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*max_weights, weights)
        elif self.weight_type == 'Volume':
            weights = (target_volume + self.eps).float().reciprocal()
            # weights = 1/(target_volume ** 2+self.eps)
            temp_weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
            max_weights = temp_weights.max(dim=1, keepdim=True)[0]
            weights = torch.where(torch.isinf(weights), torch.ones_like(weights) * max_weights, weights)
        elif self.weight_type == 'Uniform':
            weights = torch.ones(shape[0], shape[1] - int(self.no_bg))
            weights[0][4] = 4

        else:
            raise ValueError("Class weighting type {} does not exists!".format(self.weight_type))
        weights = weights / weights.max()
        # print(weights)
        weights = weights.to(source.device)
        # intersection = torch.tensor(1)
        intersection = (source_flat * target_flat).sum(2)
        # intersection = source_flat * target_flat
        # intersection = intersection.sum(2)
        scores = (2. * (intersection.float()) + self.eps) / (
                (source_volume.float() + target_volume.float()) + 2 * self.eps)

        return 1 - (weights * scores).sum() / weights.sum()

"""
Image similarity loss
"""


class NormalizedCrossCorrelationLoss(nn.Module):
    """
    The ncc loss: 1- NCC
    """

    def __init__(self):
        super(NormalizedCrossCorrelationLoss, self).__init__()

    def forward(self, input: torch.tensor, target: torch.tensor):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        input_minus_mean = input - torch.mean(input, 1, keepdim=True)
        target_minus_mean = target - torch.mean(target, 1, keepdim=True)
        nccSqr = (input_minus_mean * target_minus_mean).mean(1) / (
            torch.sqrt((input_minus_mean**2).mean(1)) * torch.sqrt((target_minus_mean**2).mean(1)))
        nccSqr = nccSqr.mean()
        return 1 - nccSqr


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input: torch.tensor, target: torch.tensor):
        return ((input - target) ** 2).mean()


class LNCCLoss(nn.Module):
    def initialize(self, kernel_sz=[9, 9, 9], voxel_weights=None):
        pass

    def __stepup(self, img_sz, use_multi_scale=True):
        max_scale = min(img_sz)
        if use_multi_scale:
            if max_scale > 128:
                self.scale = [int(max_scale / 16), int(max_scale / 8), int(max_scale / 4)]
                self.scale_weight = [0.1, 0.3, 0.6]
                self.dilation = [2, 2, 2]


            elif max_scale > 64:
                self.scale = [int(max_scale / 4), int(max_scale / 2)]
                self.scale_weight = [0.3, 0.7]
                self.dilation = [2, 2]
            else:
                self.scale = [int(max_scale / 2)]
                self.scale_weight = [1.0]
                self.dilation = [1]
        else:
            self.scale_weight = [int(max_scale / 4)]
            self.scale_weight = [1.0]
        self.num_scale = len(self.scale)
        self.kernel_sz = [[scale for _ in range(3)] for scale in self.scale]
        self.step = [[max(int((ksz + 1) / 4), 1) for ksz in self.kernel_sz[scale_id]] for scale_id in
                     range(self.num_scale)]
        self.filter = [torch.ones([1, 1] + self.kernel_sz[scale_id]).cuda() for scale_id in range(self.num_scale)]

        self.conv = F.conv3d

    def forward(self, input, target, inst_weights=None, train=None):
        self.__stepup(img_sz=list(input.shape[2:]))
        input_2 = input ** 2
        target_2 = target ** 2
        input_target = input * target
        lncc_total = 0.
        for scale_id in range(self.num_scale):
            input_local_sum = self.conv(input, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                        stride=self.step[scale_id]).view(input.shape[0], -1)
            target_local_sum = self.conv(target, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                         stride=self.step[scale_id]).view(input.shape[0],
                                                                          -1)
            input_2_local_sum = self.conv(input_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                          stride=self.step[scale_id]).view(input.shape[0],
                                                                           -1)
            target_2_local_sum = self.conv(target_2, self.filter[scale_id], padding=0, dilation=self.dilation[scale_id],
                                           stride=self.step[scale_id]).view(
                input.shape[0], -1)
            input_target_local_sum = self.conv(input_target, self.filter[scale_id], padding=0,
                                               dilation=self.dilation[scale_id], stride=self.step[scale_id]).view(
                input.shape[0], -1)

            input_local_sum = input_local_sum.contiguous()
            target_local_sum = target_local_sum.contiguous()
            input_2_local_sum = input_2_local_sum.contiguous()
            target_2_local_sum = target_2_local_sum.contiguous()
            input_target_local_sum = input_target_local_sum.contiguous()

            numel = float(np.array(self.kernel_sz[scale_id]).prod())

            input_local_mean = input_local_sum / numel
            target_local_mean = target_local_sum / numel

            cross = input_target_local_sum - target_local_mean * input_local_sum - \
                    input_local_mean * target_local_sum + target_local_mean * input_local_mean * numel
            input_local_var = input_2_local_sum - 2 * input_local_mean * input_local_sum + input_local_mean ** 2 * numel
            target_local_var = target_2_local_sum - 2 * target_local_mean * target_local_sum + target_local_mean ** 2 * numel

            lncc = cross * cross / (input_local_var * target_local_var + 1e-5)
            lncc = 1 - lncc.mean()
            lncc_total += lncc * self.scale_weight[scale_id]

        return lncc_total


class VoxelMorphLNCC(nn.Module):
    def __init__(self, filter_size=9, eps=1e-6):
        super(VoxelMorphLNCC, self).__init__()
        self.filter_size = filter_size
        self.win_numel = self.filter_size ** 3
        self.filter = nn.Parameter(torch.ones(1, 1, filter_size, filter_size, filter_size)).cuda()
        self.eps = eps

    def forward(self, I, J):
        I_square = I ** 2
        J_square = J ** 2
        I_J = I * J

        I_local_sum = F.conv3d(I, self.filter, padding=0)
        J_local_sum = F.conv3d(J, self.filter, padding=0)
        I_square_local_sum = F.conv3d(I_square, self.filter, padding=0)
        J_square_local_sum = F.conv3d(J_square, self.filter, padding=0)
        I_J_local_sum = F.conv3d(I_J, self.filter, padding=0)

        I_local_mean = I_local_sum / self.win_numel
        J_local_mean = J_local_sum / self.win_numel

        cross = I_J_local_sum - I_local_mean * J_local_sum - J_local_mean * I_local_sum + I_local_mean * J_local_mean * self.win_numel
        I_var = I_square_local_sum - 2 * I_local_mean * I_local_sum + I_local_mean ** 2 * self.win_numel
        J_var = J_square_local_sum - 2 * J_local_mean * J_local_sum + J_local_mean ** 2 * self.win_numel

        cc = (cross ** 2)/ (I_var * J_var + self.eps)

        return 1 - cc.mean()

class MIND_2D(torch.nn.Module):
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MIND_2D, self).__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma2 = gaussian_patch_sigma * gaussian_patch_sigma

        # calc shifted images in non local region
        self.image_shifter = torch.nn.Conv2d(in_channels=1, out_channels=self.nl_size * self.nl_size,
                                             kernel_size=(self.nl_size, self.nl_size),
                                             stride=1, padding=((self.nl_size-1)//2, (self.nl_size-1)//2),
                                             dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            t = torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i % self.nl_size, i//self.nl_size] = 1
            self.image_shifter.weight.data[i] = t

        # patch summation
        self.summation_patcher = torch.nn.Conv2d(in_channels=self.nl_size*self.nl_size, out_channels=self.nl_size*self.nl_size,
                                                 kernel_size=(self.p_size, self.p_size),
                                                 stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                                 dilation=1, groups=self.nl_size*self.nl_size, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            # gaussian kernel
            t = torch.zeros((1, self.p_size, self.p_size))
            cx = (self.p_size-1)//2
            cy = (self.p_size-1)//2
            for j in range(self.p_size * self.p_size):
                x = j % self.p_size
                y = j//self.p_size
                d2 = torch.norm(torch.tensor([x-cx, y-cy]).float(), 2)
                t[0, x, y] = exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] = t

        # neighbor images
        self.neighbors = torch.nn.Conv2d(in_channels=1, out_channels=self.n_size*self.n_size,
                                         kernel_size=(self.n_size, self.n_size),
                                         stride=1, padding=((self.n_size-1)//2, (self.n_size-1)//2),
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.zeros((1, self.n_size, self.n_size))
            t[0, i % self.n_size, i//self.n_size] = 1
            self.neighbors.weight.data[i] = t


        # neighbor patcher
        self.neighbor_summation_patcher = torch.nn.Conv2d(in_channels=self.n_size*self.n_size, out_channels=self.n_size * self.n_size,
                                                          kernel_size=(self.p_size, self.p_size),
                                                          stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                                          dilation=1, groups=self.n_size*self.n_size, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] = t

    def forward(self, orig_1):
        # xx = orig.shape
        orig = orig_1.squeeze(0)
        rr = orig.shape[1]
        assert(len(orig.shape) == 4)
        assert(orig.shape[1] == 1)

        # get original image channel stack
        orig_stack = torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size*self.nl_size)], dim=1)

        # get shifted images
        shifted = self.image_shifter(orig)

        # get image diff
        diff_images = shifted - orig_stack

        # diff's L2 norm
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))

        # calc neighbor's variance
        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))
        Vx = neighbor_images.var(dim=1).unsqueeze(dim=1)

        # output mind
        nume = torch.exp(-Dx_alpha / (Vx + 1e-8))
        denomi = nume.sum(dim=1).unsqueeze(dim=1)
        mind = nume / denomi
        return mind
class MIND(torch.nn.Module):
    def __init__(self, non_local_region_size=5, patch_size=3, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MIND, self).__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma2 = gaussian_patch_sigma * gaussian_patch_sigma *gaussian_patch_sigma

        # calc shifted images in non local region
        self.image_shifter = torch.nn.Conv3d(in_channels=1, out_channels=self.nl_size * self.nl_size * self.nl_size,
                                             kernel_size=(self.nl_size, self.nl_size, self.nl_size),
                                             stride=1, padding=((self.nl_size-1)//2, (self.nl_size-1)//2, (self.nl_size-1)//2),
                                             dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size*self.nl_size):
            print("i:",i)
            t = torch.zeros((1, self.nl_size, self.nl_size, self.nl_size))
            t[0, i % self.nl_size, (i - i//self.nl_size*self.nl_size)//self.nl_size, i//(self.nl_size * self.nl_size)] = 1
            self.image_shifter.weight.data[i] = t

        # patch summation
        self.summation_patcher = torch.nn.Conv3d(in_channels=self.nl_size*self.nl_size*self.nl_size, out_channels=self.nl_size*self.nl_size*self.nl_size,
                                                 kernel_size=(self.p_size, self.p_size, self.p_size),
                                                 stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2, (self.p_size-1)//2),
                                                 dilation=1, groups=self.nl_size*self.nl_size*self.nl_size, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size*self.nl_size):
            # gaussian kernel
            t = torch.zeros((1, self.p_size, self.p_size, self.p_size))
            cx = (self.p_size-1)//2
            cy = (self.p_size-1)//2
            cz = (self.p_size - 1)//2
            for j in range(self.p_size * self.p_size * self.p_size):
                x = j % self.p_size
                y = (j - j//self.p_size * self.p_size) // self.p_size
                z = j//(self.p_size*self.p_size)

                d2 = torch.norm(torch.tensor([x-cx, y-cy, z-cz]).float(), 2)
                t[0, x, y, z] = exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] = t

        # neighbor images
        self.neighbors = torch.nn.Conv3d(in_channels=1, out_channels=self.n_size*self.n_size*self.n_size,
                                         kernel_size=(self.n_size, self.n_size, self.n_size),
                                         stride=1, padding=((self.n_size-1)//2, (self.n_size-1)//2, (self.n_size-1)//2),
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size*self.n_size):
            t = torch.zeros((1, self.n_size, self.n_size, self.n_size))
            t[0, i % self.n_size, (i - i//self.n_size * self.n_size) //self.n_size, i//(self.n_size * self.n_size)] = 1
            self.neighbors.weight.data[i] = t


        # neighbor patcher
        self.neighbor_summation_patcher = torch.nn.Conv3d(in_channels=self.n_size*self.n_size*self.n_size, out_channels=self.n_size * self.n_size*self.n_size,
                                                          kernel_size=(self.p_size, self.p_size, self.p_size),
                                                          stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2, (self.p_size-1)//2),
                                                          dilation=1, groups=self.n_size*self.n_size * self.n_size, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size*self.n_size):
            t = torch.ones((1, self.p_size, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] = t

    # def forward(self, orig_1):
    #     # xx = orig.shape
    #     orig = orig_1.squeeze(0)
    def forward(self, orig):
            # xx = orig.shape
        # orig = orig_1.squeeze(0)
        # rr = orig.shape[1]
        assert(len(orig.shape) == 5)
        assert(orig.shape[1] == 1)

        # get original image channel stack
        orig_stack = torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size*self.nl_size*self.nl_size)], dim=1)

        # get shifted images
        shifted = self.image_shifter(orig)
        # cc = torch.isnan(orig_stack).any()
        # get image diff
        diff_images = shifted - orig_stack

        # diff's L2 norm
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))

        # calc neighbor's variance
        PP = self.neighbors(orig)
        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))

        # oooo = neighbor_images.var(dim=2)
        Vx = neighbor_images.var(dim=1).unsqueeze(dim=1)
        cc0 = torch.isnan(Vx).any()
        # output mind
        nume = torch.exp(-Dx_alpha / (Vx + 1e-5))
        cc1 = torch.isnan(nume).any()
        denomi = nume.sum(dim=1).unsqueeze(dim=1)
        cc2 = torch.isnan(denomi).any()
        mind = nume / (denomi + 1e-5)
        cc3 = torch.isnan(mind).any()
        return mind

class MINDLoss(torch.nn.Module):
    def __init__(self, non_local_region_size=5, patch_size=3, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MINDLoss, self).__init__()
        self.nl_size = non_local_region_size
        self.MIND = MIND(non_local_region_size=non_local_region_size,
                         patch_size=patch_size,
                         neighbor_size=neighbor_size,
                         gaussian_patch_sigma=gaussian_patch_sigma)

    def forward(self, input, target):
        in_mind = self.MIND(input)
        cc = torch.isnan(in_mind).any()
        tar_mind = self.MIND(target)
        # vv = nn.MSELoss(in_mind, tar_mind)
        mind_diff = in_mind - tar_mind
        # cc = torch.isnan(mind_diff).any()
        l1 =torch.norm(mind_diff, 1)
        return l1/(input.shape[2] * input.shape[3] * input.shape[4] * self.nl_size * self.nl_size * self.nl_size)


class SSIM3D(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 4
        self.window = self.create_window_3D(window_size, self.channel)

    def _ssim_3D(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
        # kk = mu1[:, :, 119,53,85]
        mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)
        # kko = mu2[:, :, 119,53,85]
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        # fifif = img1 * img1
        # fifif_2 = img1 * img2
        # oo = fifif[:, :, 119,53,85]
        # ll = fifif_2[:, :, 119,53,85]
        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        # dod = sigma1_sq[:, :, 119,53,85]
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        # dod1 = sigma2_sq[:, :, 119,53,85]
        # kko_ = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel)
        # kko_ = kko_[:, :, 119, 53, 85]
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        # dod2 = sigma12[:, :, 119,53,85]
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

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
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        fff = self._ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

        return self._ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)



"""
deformation field regularization loss
"""


class gradientLoss(nn.Module):
    """
    regularization loss of the spatial gradient of a 3d deformation field
    """

    def __init__(self, norm='L2', spacing=(1, 1, 1), normalize=True):
        super(gradientLoss, self).__init__()
        self.norm = norm
        self.spacing = torch.tensor(spacing).float()
        self.normalize = normalize
        if self.normalize:
            self.spacing /= self.spacing.min()

    def forward(self, input):
        """
        :param norm: 'L1' or 'L2'
        :param input: Nx3xDxHxW
        :return:
        """
        self.spacing = self.spacing.to(input.device)
        spatial_dims = torch.tensor(input.shape[2:]).float().to(input.device)
        if self.normalize:
            spatial_dims /= spatial_dims.min()

        # dx = torch.abs(input[:, :, 2:, :, :] + input[:, :, :-2, :, :] - 2 * input[:, :, 1:-1, :, :])\
        #          .view(input.shape[0], input.shape[1], -1)
        #
        # dy = torch.abs(input[:, :, :, 2:, :] + input[:, :, :, :-2, :] - 2 * input[:, :, :, 1:-1, :]) \
        #          .view(input.shape[0], input.shape[1], -1)
        #
        # dz = torch.abs(input[:, :, :, :, 2:] + input[:, :, :, :, :-2] - 2 * input[:, :, :, :, 1:-1]) \
        #          .view(input.shape[0], input.shape[1], -1)
        #
        # according to df_x = [df(x+h) - df(x-h)] /  2h
        # dx = torch.abs(input[:, :, 2:, :, :] - input[:, :, :-2, :, :]).view(input.shape[0], input.shape[1], -1)
        #
        # dy = torch.abs(input[:, :, :, 2:, :] + input[:, :, :, :-2, :]).view(input.shape[0], input.shape[1], -1)
        #
        # dz = torch.abs(input[:, :, :, :, 2:] + input[:, :, :, :, :-2]).view(input.shape[0], input.shape[1], -1)
        # if self.norm == 'L2':
        #     dx = (dx ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0])) ** 2
        #     dy = (dy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1])) ** 2
        #     dz = (dz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2])) ** 2
        # d = (dx.mean() + dy.mean() + dz.mean()) / 3.0





        dx = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])
        dy = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])
        dz = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0
        # if self.loss_mult is not None:
        #     grad *= self.loss_mult
        return grad


        #
        # return d


class BendingEnergyLoss(nn.Module):
    """
    regularization loss of bending energy of a 3d deformation field
    """

    def __init__(self, norm='L2', spacing=(1, 1, 1), normalize=True):
        super(BendingEnergyLoss, self).__init__()
        self.norm = norm
        self.spacing = torch.tensor(spacing).float()
        self.normalize = normalize
        if self.normalize:
            self.spacing /= self.spacing.min()

    def forward(self, input):
        """
        :param norm: 'L1' or 'L2'
        :param input: Nx3xDxHxW
        :return:
        """
        self.spacing = self.spacing.to(input.device)
        spatial_dims = torch.tensor(input.shape[2:]).float().to(input.device)
        if self.normalize:
            spatial_dims /= spatial_dims.min()

        # according to
        # f''(x) = [f(x+h) + f(x-h) - 2f(x)] / h^2
        # f_{x, y}(x, y) = [df(x+h, y+k) + df(x-h, y-k) - df(x+h, y-k) - df(x-h, y+k)] / 2hk

        ddx = torch.abs(input[:, :, 2:, 1:-1, 1:-1] + input[:, :, :-2, 1:-1, 1:-1] - 2 * input[:, :, 1:-1, 1:-1, 1:-1])\
                 .view(input.shape[0], input.shape[1], -1)

        ddy = torch.abs(input[:, :, 1:-1, 2:, 1:-1] + input[:, :, 1:-1, :-2, 1:-1] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]) \
                 .view(input.shape[0], input.shape[1], -1)

        ddz = torch.abs(input[:, :, 1:-1, 1:-1, 2:] + input[:, :, 1:-1, 1:-1, :-2] - 2 * input[:, :, 1:-1, 1:-1, 1:-1]) \
                 .view(input.shape[0], input.shape[1], -1)

        dxdy = torch.abs(input[:, :, 2:, 2:, 1:-1] + input[:, :, :-2, :-2, 1:-1] -
                         input[:, :, 2:, :-2, 1:-1] - input[:, :, :-2, 2:, 1:-1]).view(input.shape[0], input.shape[1], -1)

        dydz = torch.abs(input[:, :, 1:-1, 2:, 2:] + input[:, :, 1:-1, :-2, :-2] -
                         input[:, :, 1:-1, 2:, :-2] - input[:, :, 1:-1, :-2, 2:]).view(input.shape[0], input.shape[1], -1)

        dxdz = torch.abs(input[:, :, 2:, 1:-1, 2:] + input[:, :, :-2, 1:-1, :-2] -
                         input[:, :, 2:, 1:-1, :-2] - input[:, :, :-2, 1:-1, 2:]).view(input.shape[0], input.shape[1], -1)


        if self.norm == 'L2':
            ddx = (ddx ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0]**2)) ** 2
            ddy = (ddy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1]**2)) ** 2
            ddz = (ddz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2]**2)) ** 2
            dxdy = (dxdy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0] * self.spacing[1])) ** 2
            dydz = (dydz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1] * self.spacing[2])) ** 2
            dxdz = (dxdz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2] * self.spacing[0])) ** 2

        d = (ddx.mean() + ddy.mean() + ddz.mean() + 2*dxdy.mean() + 2*dydz.mean() + 2*dxdz.mean()) / 9.0
        return d


class L2Loss(nn.Module):

    def forward(self, input):
        return (input ** 2).mean()


class MyL2Loss(nn.Module):

    def forward(self, pred, gt):
        loss = (torch.sum(pred) / torch.sum(gt) - 1) ** 2
        loss += torch.mean((pred / gt - 1) ** 2)
        return loss


class MSLELoss(nn.Module):
    def forward(self, warped, gt):
        return ((torch.log(warped + 1) - torch.log(gt + 1)) ** 2).mean()


def euclidean(y_true, y_pred):
    sim = np.sqrt(np.sum(np.square(y_true - y_pred)))
    return sim

def MAE(y_true, y_pred):
    mse = np.mean(np.square(y_true - y_pred))
    return mse

def huber_loss(y_true, y_pred):
    return F.huber_loss(y_pred,y_true)

class dist_loss(nn.Module):
    def forward(self,y_true, y_pred):
        loss = (y_true - y_pred) * (y_true - y_pred) / (
                    y_true.shape[0] * y_true.shape[1] * y_true.shape[2] * y_true.shape[3] * y_true.shape[4])
        return loss.sum()

class color_loss(nn.Module):
    def forward(self, y_true, y_pred, y):
        similarity = torch.cosine_similarity(y_true, y_pred, dim=1)
        similarity[y == 0] = 1
        similarity = 1 - similarity
        return torch.sum(similarity) / (y_true.shape[0] * y_true.shape[2] * y_true.shape[3] * y_true.shape[4])



loss_dict = {
    'ncc': NormalizedCrossCorrelationLoss,
    'lncc': VoxelMorphLNCC,
    'mse': nn.MSELoss,
    'gradient': gradientLoss,
    'bendingEnergy': BendingEnergyLoss,
    'dice': DiceLossMultiClass,
    'L2': L2Loss,
    'focal': FocalLoss,
    'cross_entropy': nn.CrossEntropyLoss,
    'soft_cross_entropy': SoftCrossEntropy,
    'smoothl1loss': nn.HuberLoss,
    'maeloss': nn.L1Loss,
    'msleloss': MSLELoss,
    'ssim': SSIM3D,
    'Huberloss': F.huber_loss,
    'euclidean': euclidean,
    'dist_loss': dist_loss,
    'color_loss': color_loss,
    'mse_joint':MSELossMultiClass,
    'mind':MINDLoss





}


def get_loss_function(loss_name):
    if loss_name in get_available_losses():
        return loss_dict[loss_name]
    else:
        raise KeyError("Network {} is not available!\n Choose from: {}".format(loss_name, get_available_losses()))


def get_available_losses():
    return loss_dict.keys()


class NaivePenalty(nn.Module):
    def __init__(self, boundary=[], class_ind=[], total=0):
        super(NaivePenalty, self).__init__()
        self.bound = boundary
        self.ind = class_ind
        self.total = total

    def forward(self, soft_pred):
        bound_id = 0
        loss = 0
        for ind in self.ind:
            pre_i = soft_pred[ind]
            lower = self.bound[bound_id]
            upper = self.bound[bound_id + 1]
            bound_id += 2
            val = torch.sum(pre_i)
            if val < lower:
                too_small = (lower - val) ** 2
                loss += too_small
            elif val > upper:
                too_big = (val - upper) ** 2
                loss += too_big
        loss /= self.total
        loss /= len(self.ind)
        return loss



if __name__ == '__main__':
    input = torch.randn(2, 1, 160, 160, 160)
    target = torch.randn(2, 1, 160, 160, 160)
    criterion2 = NormalizedCrossCorrelationLoss()
    loss2 = criterion2(input, input)
