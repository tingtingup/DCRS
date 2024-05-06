#!/usr/bin/env python
"""
Created by zhenlinx on 11/4/18
last modified by oylei98@163.com on 2021/10/13
"""

import os
import numpy as np
import SimpleITK as sitk


import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math

from .utils import get_identity_transform_batch

n_classes = 10


class SegDataSetMindBoggle(Dataset):
    """
    seg dataset for MindBoggle101
        kind 0:['gt']                       (val or test or gen)
        kind 1:['gt'] or ['gt', 'flip']   (full supervised)

        ------------------------------ papers' weak supervised mode ------------------------------------------
        kind 4:['weak'] or ['weak', 'flip']   (DeepAtlas only use weak data)
    """

    def __init__(self, txt_files, data_dir, data_kind=[]):
        # data settings
        self.data_dir = data_dir
        self.image_list, self.segmentation_list, self.name_list, self.field_list = self.read_image_segmentation_list(
            txt_files)

        self.data_kind = data_kind
        self.data_kind_no_flip = data_kind
        # check data
        if len(self.image_list) != len(self.segmentation_list):
            raise ValueError("The numbers of images and segmentations are different")
        self.length = len(self.image_list)

        self.flip = False
        self.kind_len = len(self.data_kind)
        if 'flip' in self.data_kind:
            self.flip = True
            self.kind_len -= 1
            self.data_kind_no_flip.remove('flip')

        self.debug = False
        # -----------------preload atlas img, seg, seg_hot, cause they are frequently used----------------------
        atlas = np.load('../data/process/NKI-RS-22_volumes/NKI-RS-22-10/brain.npy')
        atlas = atlas[np.newaxis, np.newaxis, ...]
        self.atlas = torch.from_numpy(atlas)
        atlas_seg = np.load('../data/process/NKI-RS-22_volumes/NKI-RS-22-10/mask.npy')
        atlas_seg_hot = np.eye(n_classes)[atlas_seg].transpose(3, 0, 1, 2)
        atlas_seg_hot = atlas_seg_hot[np.newaxis, ...]
        self.atlas_seg = torch.from_numpy(atlas_seg)
        self.atlas_seg_hot = torch.from_numpy(atlas_seg_hot)

    def __len__(self):
        data_len = self.kind_len * self.length
        return 2 * data_len if self.flip else data_len

    def __getitem__(self, id):
        sample = self.get_sample(id)

        return [item for item in sample.values()]

    def get_sample(self, id):
        # id process
        id_ori = id

        # setting id > mid means flip
        mid = int(self.__len__() / 2)
        self.flip_flag = False
        if self.flip and id_ori >= mid:
            self.flip_flag = True
            id -= mid

        mod = id // self.length
        id %= self.length
        if self.debug:
            print('id_ori is:', id_ori)
            print('id is:', id)
        img_name = self.name_list[id]

        # ------------------------------- assign image and mask dir ------------------------------------------------
        if id == 0:
            if self.flip_flag:
                mov_file_name = self.image_list[id].replace('mindboggle', 'process').replace('brain', 'brain_flip')
                seg_file_name = self.segmentation_list[id].replace('mindboggle', 'process').replace('mask', 'mask_flip')
            else:
                mov_file_name = self.image_list[id].replace('mindboggle', 'process')
                seg_file_name = self.segmentation_list[id].replace('mindboggle', 'process')

        elif not id == 0:

            data_kind = self.data_kind_no_flip[mod]
            if data_kind == 'gt':
                img_dir = 'process'
                mask_dir = 'process'

            elif data_kind == 'weak':
                img_dir = 'process'
                mask_dir = 'weakly_data'

            # elif data_kind == 'aug':
            #     img_dir = 'augmentation_data'
            #     mask_dir = 'weakly_data'

            else:
                raise ValueError('cannot decided which dir to read image and mask in dataset.py', self.data_kind,
                                 data_kind)

            if self.flip_flag:
                mov_file_name = self.image_list[id].replace('mindboggle', img_dir).replace('brain', 'brain_flip')
                seg_file_name = self.segmentation_list[id].replace('mindboggle', mask_dir).replace('mask', 'mask_flip')
            else:
                mov_file_name = self.image_list[id].replace('mindboggle', img_dir)
                seg_file_name = self.segmentation_list[id].replace('mindboggle', mask_dir)

        else:
            raise ValueError('cannot assign mov and seg file name', self.data_kind)

        # load sample, input id for getting data kind
        if self.debug:
            print(id, id_ori, img_name, mov_file_name, seg_file_name)
        sample = self.load_sample(img_name, mov_file_name, seg_file_name)

        return sample

    # main functions realized here
    def load_sample(self, name, mov_file_name, seg_file_name):
        """
        Load a segmentation data sample into a dictionary
        """

        # check existing
        if mov_file_name and not os.path.exists(mov_file_name):
            raise ValueError(mov_file_name + ' not exist!')
        if seg_file_name and not os.path.exists(seg_file_name):
            raise ValueError(seg_file_name + ' not exist!')

        sample = {}
        if mov_file_name:
            mov_img = np.load(mov_file_name)
            img = mov_img[np.newaxis, ...]
            img = torch.from_numpy(img)
            # make sure img is 4-dimension
            sample['image'] = img if len(img.shape) <= 4 else img[0, ...]

        if seg_file_name:
            seg = np.load(seg_file_name)  # seg intensity from 0 to 16
            # seg_name is ./mask_flip.npy or ./mask.npy, seg hot is ./mask_flip_hot.npy or ./mask_hot.npy
            seg_hot_name = list(seg_file_name)
            seg_hot_name.insert(-4, '_hot')
            seg_hot_name = ''.join(seg_hot_name)
            seg_hot = np.load(seg_hot_name)

            seg = torch.from_numpy(seg)
            seg_hot = torch.from_numpy(seg_hot)
            sample['segmentation'] = seg
            sample['name'] = name
            # make sure seg_one-hot is 4-dimension
            sample['segmentation_onehot'] = seg_hot if len(seg_hot.shape) <= 4 else seg_hot[0, ...]

        return sample

    # get image dir from txt file
    @staticmethod
    def read_image_segmentation_list(text_files):
        """
        read txt file contents and to image_list segmentation_list and name_list ang then return them
        """
        image_list = []
        segmentation_list = []
        name_list = []
        deform_field_list = []
        if isinstance(text_files, str):
            text_files = [text_files]
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    image_name = line.strip("\n")
                    image_name = image_name.strip('/home/qulab/disk/OYL/data/mindboggle/')

                    # todo: not robust
                    # handling OASIS missing 'O' issue
                    if image_name[0] == 'A':
                        image_name = 'O' + image_name

                    name_list.append(image_name)
                    image_list.append(line.strip("\n") + '/brain.npy')
                    segmentation_list.append(line.strip("\n") + '/mask.npy')
                    deform_field_list.append(line.strip("\n") + '/field.npy')

        return image_list, segmentation_list, name_list, deform_field_list


class RegDataSetMindBoggle(Dataset):
    """
    dataset for image registration
    """

    def __init__(self, txt_files, data_dir, data_kind=[]):
        self.data_dir = data_dir
        self.data_kind = data_kind
        self.image_list, self.name_list, self.segmentation_list, self.ori_img_list = self.read_image_segmentation_list(
            txt_files)
        self.length = len(self.image_list)

        # train a mov to atlas reg net for appearance transform
        # self.inverse = inverse

    def __len__(self):
        mul = 1
        if 'flip' in self.data_kind:
            mul = 2
        return mul * self.length

    def __getitem__(self, id):
        # id = 0 as atlas
        # atlas to subject, i.e. id 0 warp to others

        # TODO: para 50 not robust, it used here for judge whether we are training
        if self.length > 50 and id == 0:
            fixed_ind = int(np.random.randint(1, self.length - 1, 1))
        else:
            fixed_ind = id

        # sample1 will warp to sample2
        source = self.get_atlas()
        target = self.get_sample(fixed_ind)
        # in order image, seg, name, seg_onehot(optional)
        # if self.inverse:
        #     return [item for item in target.values()], [item for item in source.values()]
        # else:
        return [item for item in source.values()], [item for item in target.values()]

    def get_sample(self, id):
        id_ori = id
        id %= self.length
        image_file_name = self.image_list[id]
        segmentation_file_name = self.segmentation_list[id] if 'with_seg' in self.data_kind else None

        image_name = self.name_list[id]
        ori_img_name = self.ori_img_list[id]
        # image_file_name = image_file_name.replace('mindboggle', 'process')
        # if segmentation_file_name:
        #     if 'gt' in self.data_kind:
        #         segmentation_file_name = segmentation_file_name.replace('mindboggle', 'process')
        #     else:
        #         segmentation_file_name = segmentation_file_name.replace('mindboggle', 'pseudo_by_seg')
        # print (segmentation_file_name)
        sample = self.load_sample(id_ori, image_name, image_file_name, segmentation_file_name,ori_img_name)
        # sample = self.load_sample(id_ori, image_name, image_file_name, ori_img_name)

        return sample

    def load_sample(self, id_ori, name, img_file_name, seg_file_name, ori_img):
        """
        Load a segmentation data sample into a dictionary
        """
        # check existing
        if not os.path.exists(img_file_name):
            raise ValueError(img_file_name + ' not exist!')
        # if seg_file_name and not os.path.exists(seg_file_name):
        #     raise ValueError(seg_file_name + ' not exist!')
        if ori_img and not os.path.exists(ori_img):
            raise ValueError(ori_img + ' not exist!')

        mod = id_ori // self.length
        sample = {}
        # print("name:", img_file_name)
        img = np.load(img_file_name)
        xxxx = img.dtype
        ori_img = np.load(ori_img)

        bb = ori_img[67,:,:]
        hhhhhh = ori_img.dtype
        img_1 = img[np.newaxis, ...]
        ori_img_1 = ori_img[np.newaxis, ...]
        img_cat = np.concatenate((img_1, ori_img_1), 0)
        cc = img.shape
        if 'cat' in self.data_kind:
            img_file_name_ = img_file_name.replace('indice', 'edt')
            img_ = np.load(img_file_name_)
            img_ = img_[np.newaxis, ...]
            ori_img = ori_img[np.newaxis, ...]
            cc_ = img_.shape
            cc_1 = img.shape
            # print("_-----", cc_)
            # print("******", cc_1)
            img_cat = np.concatenate((img, img_), 0)
            ori_img = torch.from_numpy(ori_img)
        if mod == 1:
            img = np.flip(img, 2)
        if 'indice' in self.data_kind and 'cat' in self.data_kind:
            img = torch.from_numpy(img_cat)
        elif 'indice' in self.data_kind and 'cat' not in self.data_kind:
            img = torch.from_numpy(img)
        else:
            img = np.exp(-img)
            # img_save = sitk.GetImageFromArray(img, isVector=False)
            # dir_test_disp_field = '/home/amax/disk/tthan/e'
            # save_name_nii = name + '.nii.gz'
            # savepath_disp_field_niigz = dir_test_disp_field + '/' + save_name_nii
            # sitk.WriteImage(img_save, savepath_disp_field_niigz)
            img = img[np.newaxis, ...]
            ori_img = ori_img[np.newaxis, ...]
            img = torch.from_numpy(img)
            ori_img = torch.from_numpy(ori_img)
            cat_ori_img_img = torch.from_numpy(img_cat)

        if seg_file_name:
            print ("seg:", seg_file_name)
            seg = np.load(seg_file_name)  # seg intensity from 0 to 16
            # seg_hot = np.eye(n_classes)[seg].transpose(3, 0, 1, 2)
            seg_hot = np.load(seg_file_name.replace('mask', 'mask_hot'))
            # seg_hot = seg_hot[np.newaxis, ...]
            seg = seg[np.newaxis, ...]
            seg = torch.from_numpy(seg)
            seg_hot = torch.from_numpy(seg_hot)

        sample['image'] = img
        sample['ori_image'] = ori_img
        if seg_file_name:
            sample['segmentation'] = seg
        sample['name'] = name
        if seg_file_name:
            sample['segmentation_onehot'] = seg_hot
        # sample['cat'] = cat_ori_img_img
        return sample

    def get_atlas(self):
        # image_file_name = '/home/amax/disk/tthan/ori_mini_code_from_oyl/data/process/NKI-RS-22_volumes/NKI-RS-22-10/brain.npy'
        # # image_file_name = '/home/amax/disk/tthan/SDF/edt/NKI-RS-22_volumes_NKI-RS-22-10_mask_hot.npy'
        # # image_file_name = '/home/amax/disk/tthan/SDF/indice/NKI-RS-22_volumes_NKI-RS-22-10_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf/edt/CCF_CCF_roi_resample_mask_hot.npy'
        image_file_name = '/home/amax/disk/tthan/SDF/ccf_resample/edt/ccf_resample_CCF_roi_resample_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/visor_atlas/edt/visor_new_resample_VISOR_atlas_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/fmost_ants_affine_resample/edt/Fmost_Affine_ants_resample_17788_norm_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf_resample/edt_bg_nobg/ccf_resample_CCF_roi_resample_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf_resample/edt_bg_negative/ccf_resample_CCF_roi_resample_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf_resample/edt/ccf_resample_CCF_roi_resample_mask_hot_normalization_global.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf_resample/edt/ccf_resample_CCF_roi_resample_mask_hot_normalization-1_global.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf_resample/edt/ccf_resample_CCF_roi_resample_mask_hot_normalization.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf_resample/edt/ccf_resample_CCF_roi_resample_mask_hot_normalization-1.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/fmost_ants_affine_resample/edt/Fmost_Affine_ants_resample_17787_norm_mask_hot.npy'
        # segmentation_file_name = '/home/amax/disk/tthan/dataset/Fmost_Affine_ants_resample/17788_norm/mask.npy' if 'with_seg' in self.data_kind else None
        # ORI_image_file_name = '/home/amax/disk/tthan/dataset/Fmost_Affine_ants_resample/17788_norm/brain.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf/edt_1/CCF_CCF_roi_resample_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf/edt/CCF_CCF_roi_resample_mask_hot_normalization.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf/edt/CCF_CCF_roi_resample_mask_hot_normalization_.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/fmost/edt/Fmost_17788_norm_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf/indice/CCF_CCF_roi_resample_mask_hot.npy'
        # image_file_name = '/home/amax/disk/tthan/SDF/ccf/indice_1/CCF_CCF_roi_resample_mask_hot.npy'
        # segmentation_file_name = image_file_name.replace('brain', 'mask') if 'with_seg' in self.data_kind else None
        # segmentation_file_name = '/home/qulab/disk/oyl/data/process/NKI-RS-22_volumes/NKI-RS-22-10/mask.npy' if 'with_seg' in self.data_kind else None
        # segmentation_file_name = '/home/amax/disk/tthan/dataset/CCF/CCF_roi_resample/mask.npy' if 'with_seg' in self.data_kind else None
        # ORI_image_file_name = '/home/amax/disk/tthan/dataset/CCF/CCF_roi_resample/brain.npy'
        # segmentation_file_name = '/home/amax/disk/tthan/dataset/VISOR_atlas/visor_new_resample/VISOR_atlas/mask.npy' if 'with_seg' in self.data_kind else None
        # ORI_image_file_name = '/home/amax/disk/tthan/dataset/VISOR_atlas/visor_new_resample/VISOR_atlas/brain.npy'
        # ORI_image_file_name = '/home/amax/disk/tthan/dataset/CCF/CCF_roi_resample/brain.npy'
        segmentation_file_name = '/home/amax/disk/tthan/dataset/CCF/ccf_resample/CCF_roi_resample/mask.npy' if 'with_seg' in self.data_kind else None
        # ORI_image_file_name = '/home/amax/disk/tthan/dataset/CCF/ccf_resample/CCF_roi_resample/brain_normalization.npy'
        ORI_image_file_name = '/home/amax/disk/tthan/dataset/CCF/ccf_resample/CCF_roi_resample/brain_normalization.npy'
        # segmentation_file_name = '/home/amax/disk/tthan/dataset/Fmost/17788_norm/mask.npy' if 'with_seg' in self.data_kind else None
        # ORI_image_file_name = '/home/amax/disk/tthan/dataset/Fmost/17788_norm/brain.npy'
        # image_name = image_file_name.strip('/home/amax/disk/tthan/ori_mini_code_from_oyl/data/process')
        # image_name = image_name.strip('/brain.npy')
        image_name = image_file_name.split('/')[-1].split('.')[0]
        # image_file_name = image_file_name.replace('mindboggle', 'process')
        # if segmentation_file_name:
        #     segmentation_file_name = segmentation_file_name.replace('mindboggle', 'process')
        sample = self.load_sample(0, image_name, image_file_name, segmentation_file_name, ORI_image_file_name)

        return sample

    @staticmethod
    def read_image_segmentation_list(text_files):
        """
        read txt file contents and to image_list segmentation_list and name_list ang then return them
        """
        image_list = []
        segmentation_list = []
        name_list = []
        ori_img_list = []
        deform_field_list = []
        # segmentation_path = '/home/qulab/disk/oyl/data/process'
        segmentation_path = '/home/amax/disk/tthan/dataset/Fmost_Affine_ants_resample'
        # segmentation_path = '/home/amax/disk/tthan/dataset/visor_new_resample_affine_'
        # segmentation_path = '/home/amax/disk/tthan/dataset/Fmost_Affine_ants_affine_to_visor'
        # segmentation_path = '/home/amax/disk/tthan/dataset/visor_new_resample_affine'
        # segmentation_path = '/home/amax/disk/tthan/dataset/Fmost'
        # segmentation_path = '/home/amax/disk/tthan/dataset/Visor'
        if isinstance(text_files, str):
            text_files = [text_files]
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    image_name = line.strip("\n")
                    image_name = image_name.split('/')[-1].split('.')[0]
                    # image_name = image_name.strip('/home/amax/disk/tthan/ori_mini_code_from_oyl/data/process/')
                    # handling OASIS missing 'O' issue
                    # if image_name[0] == 'A':
                    #     image_name = 'O' + image_name

                    name_list.append(image_name)

                    # image_list.append(line.strip("\n") + '/brain.npy')
                    image_list.append(line.strip("\n"))

                    # seg_path_split = image_name.split('_')
                    # # agencyinfo_data = seg_path_split[0] + '_' + seg_path_split[1]
                    # agencyinfo_data = seg_path_split[0]
                    # # print (seg_path_split)
                    # individualinfo_data = seg_path_split[1] + '_' + seg_path_split[2]
                    # segmentation_file_path = segmentation_path + '/' + agencyinfo_data + '/' + individualinfo_data + '/mask.npy'
                    ####fmost
                    if 'affine' in image_name:
                        segmentation_file_path = '/home/amax/disk/tthan/dataset/Fmost_Affine_ants_resample/Atlas/Atlas/affine/brain/mask.npy'
                        ori_img_file_path = '/home/amax/disk/tthan/dataset/Fmost_Affine_ants_resample/Atlas/Atlas/affine/brain/brain.npy'
                        segmentation_list.append(segmentation_file_path)
                        ori_img_list.append(ori_img_file_path)
                    else:
                        print ("11",image_name)
                        individualinfo_data = image_name.split('_')[4] + "_norm"
                        segmentation_file_path = segmentation_path + '/' + individualinfo_data + '/mask.npy'
                        # ori_img_file_path = segmentation_path + '/' + individualinfo_data + '/brain_normalization.npy'
                        ori_img_file_path = segmentation_path + '/' + individualinfo_data + '/brain_normalization.npy'
                        #############
                        # image_name = image_name.lstrip("Visor_")
                        # image_name = image_name.rstrip("_mask_hot")
                        # image_name = image_name.lstrip("visor_new_resample_affine_")
                        # image_name = image_name.rstrip("_mask_hot")
                        # individualinfo_data = image_name.split('_')[4] + "_norm"
                        # segmentation_file_path = segmentation_path + '/' + image_name + '/mask.npy'
                        # ori_img_file_path = segmentation_path + '/' + image_name + '/brain.npy'
                        segmentation_list.append(segmentation_file_path)
                        ori_img_list.append(ori_img_file_path)
                    # deform_field_list.append(line.strip("\n") + '/field.npy')
        #
        # return image_list, segmentation_list, name_list, deform_field_list
        return image_list, name_list, segmentation_list, ori_img_list

if __name__ == '__main__':
    print(os.path.realpath("../.."))
