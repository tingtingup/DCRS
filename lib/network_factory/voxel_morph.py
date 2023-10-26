#!/usr/bin/env python
"""
registration network described in voxelmorphm
Created by zhenlinx on 11/8/18
last modified by oylei98@163.com on 2021/10/13
"""

import os
import sys
import numpy as np

sys.path.append(os.path.realpath('../..'))

import torch
import torch.nn as nn
from lib.network_factory.modules import convBlock
from lib.utils import get_identity_transform_batch
import torch.nn.functional as F


class VoxelMorphCVPR2018(nn.Module):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param input_channel: channels of input data (2 for a pair of images)
    :param output_channel: channels of output data (3 for 3D registration)
    :param enc_filters: list of encoder filters. values represent the number of filters of each layer
           e.g. (16, 32, 32, 32, 32)
    :param dec_filters: list of decoder filters.
    """
    # def __init__(self, input_channel=2, output_channel=3, enc_filters=(16, 32, 32, 32, 32),
    #              dec_filters=(32, 32, 32, 8, 8)):
    def __init__(self, input_channel=2, output_channel=3, enc_filters=(16, 32, 32, 32),
                     dec_filters=(32, 32, 32, 32, 32, 16, 16)):
        super(VoxelMorphCVPR2018, self).__init__()


        self.input_channel = input_channel
        self.output_channel = output_channel
        self.enc_filters = enc_filters
        self.dec_filters = dec_filters

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsampling = nn.Upsample(scale_factor=2, mode="trilinear")

        for i in range(len(enc_filters)):
            if i == 0:
                self.encoders.append(convBlock(input_channel, enc_filters[i], stride=1, bias=True))
            else:
                self.encoders.append(convBlock(enc_filters[i-1], enc_filters[i], stride=2, bias=True))

        for i in range(len(dec_filters)):
            if i == 0:
                self.decoders.append(convBlock(enc_filters[-1], dec_filters[i], stride=1, bias=True))
            elif i < 4:
                self.decoders.append(convBlock(dec_filters[i-1] if i == 4 else dec_filters[i - 1] + enc_filters[4-i],
                                            dec_filters[i], stride=1, bias=True))
            # elif i < 3:
            #     self.decoders.append(
            #         convBlock(dec_filters[i - 1] if i == 3 else dec_filters[i - 1] + enc_filters[3 - i],
            #                   dec_filters[i], stride=1, bias=True))nn.init.xavier_normal_(m.weight.data)
            else:
                self.decoders.append(convBlock(dec_filters[i-1], dec_filters[i], stride=1, bias=True))

        self.flow = nn.Conv3d(dec_filters[-1] + enc_filters[0], output_channel, kernel_size=3, stride=1, padding=1, bias=True)

        # identity transform for computing displacement
        self.id_transform = None

    def forward(self, source, target):
        ssss = source.shape
        tttt = target.shape
        # print(ssss)
        # print(tttt)
        # target = F.interpolate(target, size=source.shape[2:], mode='trilinear')
        # diff = target - source
        # diff = source - target
        ori = torch.cat((source, target), dim=1)
        # combin = torch.cat((diff, ori), dim=1)
        # x_enc_1 = self.encoders[0](torch.cat((source, target), dim=1))
        # x_enc_1 = self.encoders[0](diff)
        # # del input
        # x_enc_2 = self.encoders[1](x_enc_1)
        # x_enc_3 = self.encoders[2](x_enc_2)
        # x_enc_4 = self.encoders[3](x_enc_3)
        # x_enc_5 = self.encoders[4](x_enc_4)
        #
        # x_dec_1 = self.decoders[0](F.interpolate(x_enc_5, size=x_enc_4.shape[2:]))
        # del x_enc_5
        # x_dec_2 = self.decoders[1](F.interpolate(torch.cat((x_dec_1, x_enc_4), dim=1), size=x_enc_3.shape[2:]))
        # del x_dec_1, x_enc_4
        # x_dec_3 = self.decoders[2](F.interpolate(torch.cat((x_dec_2, x_enc_3), dim=1), size=x_enc_2.shape[2:]))
        # del x_dec_2, x_enc_3
        # x_dec_4 = self.decoders[3](torch.cat((x_dec_3, x_enc_2), dim=1))
        # del x_dec_3, x_enc_2
        # x_dec_5 = self.decoders[4](F.interpolate(x_dec_4, size=x_enc_1.shape[2:]))
        # del x_dec_4
        # # uu = x_enc_1.shape
        # # vv = x_dec_5.shape
        # # gg = torch.cat((x_dec_5, x_enc_1),dim=1)
        # # tt = gg.shape
        # disp_field = self.flow(torch.cat((x_dec_5, x_enc_1), dim=1))
        # del x_dec_5, x_enc_1
        # aa = disp_field.shape
   ####
        # ssss = source.shape
        # tttt = target.shape
        # target = F.interpolate(target, size=source.shape[2:], mode='trilinear')
        # x_enc_1 = self.encoders[0](torch.cat((source, target), dim=1))
        # # del input
        # x_enc_2 = self.encoders[1](x_enc_1)
        # x_enc_3 = self.encoders[2](x_enc_2)
        # x_enc_4 = self.encoders[3](x_enc_3)
        # # x_enc_5 = self.encoders[4](x_enc_4)
        #
        # x_dec_1 = self.decoders[0](F.interpolate(x_enc_4, size=x_enc_3.shape[2:]))
        # del x_enc_4
        # x_dec_2 = self.decoders[1](F.interpolate(torch.cat((x_dec_1, x_enc_3), dim=1), size=x_enc_2.shape[2:]))
        # del x_dec_1, x_enc_3
        # # x_dec_3 = self.decoders[2](F.interpolate(torch.cat((x_dec_2, x_enc_3), dim=1), size=x_enc_2.shape[2:]))
        # # del x_dec_2, x_enc_3
        # x_dec_3 = self.decoders[2](torch.cat((x_dec_2, x_enc_2), dim=1))
        # del x_dec_2, x_enc_2
        # x_dec_4 = self.decoders[3](F.interpolate(x_dec_3, size=x_enc_1.shape[2:]))
        # del x_dec_3
        # # uu = x_enc_1.shape
        # # vv = x_dec_5.shape
        # # gg = torch.cat((x_dec_5, x_enc_1),dim=1)
        # # tt = gg.shape
        # disp_field = self.flow(torch.cat((x_dec_4, x_enc_1), dim=1))
        # del x_dec_4, x_enc_1
        # aa = disp_field.shape
####

        # bb = len(self.encoders)
        # ssss = source.shape
        # tttt = target.shape
        # target = F.interpolate(target, size=source.shape[2:], mode='trilinear')
        # x_enc_1 = self.encoders[0](torch.cat((source, target), dim=1))
        # x_enc_1 = self.encoders[0](torch.cat((source, target), dim=1))
        x_enc_1 = self.encoders[0](ori)
        # del input
        x_enc_2 = self.encoders[1](x_enc_1)
        x_enc_3 = self.encoders[2](x_enc_2)
        x_enc_4 = self.encoders[3](x_enc_3)
        x_enc_5 = self.encoders[4](x_enc_4)

        # x_dec_1 = self.decoders[0](F.interpolate(x_enc_5, size=x_enc_4.shape[2:]))
        # del x_enc_5
        # x_dec_2 = self.decoders[1](F.interpolate(torch.cat((x_dec_1, x_enc_4), dim=1), size=x_enc_3.shape[2:]))
        # del x_dec_1, x_enc_4
        # x_dec_3 = self.decoders[2](F.interpolate(torch.cat((x_dec_2, x_enc_3), dim=1), size=x_enc_2.shape[2:]))
        # del x_dec_2, x_enc_3
        # x_dec_4 = self.decoders[3](torch.cat((x_dec_3, x_enc_2), dim=1))
        # del x_dec_3, x_enc_2
        # x_dec_5 = self.decoders[4](F.interpolate(x_dec_4, size=x_enc_1.shape[2:]))
        # del x_dec_4
        x_dec_1 = self.decoders[0](x_enc_5)
        x_dec_1 = F.interpolate(x_dec_1, size=x_enc_4.shape[2:])
        del x_enc_5
        x_dec_2 = self.decoders[1](torch.cat((x_dec_1, x_enc_4), dim=1))
        x_dec_2 = F.interpolate(x_dec_2, size=x_enc_3.shape[2:])
        del x_dec_1, x_enc_4
        x_dec_3 = self.decoders[2](torch.cat((x_dec_2, x_enc_3), dim=1))
        x_dec_3 = F.interpolate(x_dec_3, size=x_enc_2.shape[2:])
        del x_dec_2, x_enc_3
        x_dec_4 = self.decoders[3](torch.cat((x_dec_3, x_enc_2), dim=1))
        del x_dec_3, x_enc_2
        x_dec_5 = self.decoders[4](F.interpolate(x_dec_4, size=x_enc_1.shape[2:]))
        # x_dec_5 = self.decoders[4](x_dec_4)
        # x_dec_5 = F.interpolate(x_dec_5, size=x_enc_1.shape[2:])
        del x_dec_4
        # uu = x_enc_1.shape
        # vv = x_dec_5.shape
        # gg = torch.cat((x_dec_5, x_enc_1),dim=1)
        # tt = gg.shape
        disp_field = self.flow(torch.cat((x_dec_5, x_enc_1), dim=1))
        # dd = disp_field[0, 1, 134, 100, 162]
        # jj = -disp_field[0, 1, 134, 100, 162]
        del x_dec_5, x_enc_1
        # aa = disp_field.shape
        # disp_field = disp_field * para
        # itit = disp_field[0,0,1,:,:]

        #
        # tttttt = source.shape
        if self.id_transform is None:
            self.id_transform = get_identity_transform_batch(source.shape, normalize=False).to(disp_field.device)

            #
            # self.id_transform = get_identity_transform_batch(source.shape).to(disp_field.device)
        # popop = self.id_transform
        # bb = self.id_transform.shape
        # suuuuuuu = self.id_transform[0,1,:,:]
        deform_field = disp_field + self.id_transform
        shape = disp_field.shape[2:]
        for i in range(len(shape)):
            deform_field[:, i, ...] = 2 * (deform_field[:, i, ...] / (shape[i] - 1) - 0.5)
        # mnmnm = deform_field[0,0,1,:,:]
        # print (mnmnm)
        # deform_field_ = -disp_field + self.id_transform
        # uu = deform_field[0, 1, 134, 100, 162]
        # oo = deform_field_[0, 1, 134, 100, 162]
        # s_deform = deform_field.shape
        # ff = deform_field.permute([0, 2, 3, 4, 1])
        # ll = source.shape
        # transform images
        # warped_source = F.grid_sample(source, grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
        #                               padding_mode='zeros', align_corners=True)
        return disp_field, deform_field

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                    # nn.init.xavier_uniform_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()


def demo_test():
    cuda = torch.device('cuda:1')

    # unet = UNet_light2(2,3).to(cuda)
    net = VoxelMorphCVPR2018().to(cuda)
    net.weights_init()
    print(net)
    with torch.enable_grad():
        input1 = torch.randn(1, 1, 200, 200, 160).to(cuda)
        input2 = torch.randn(1, 1, 200, 200, 160).to(cuda)
        disp_field, warped_input1, deform_field = net(input1, input2)
    pass


if __name__ == '__main__':
    demo_test()
