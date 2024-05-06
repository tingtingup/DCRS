#!/usr/bin/env python
import os
import numpy as np
import torch
from torch.utils.data import Dataset
n_classes = 9
class SegDataSet(Dataset):
    def __init__(self, txt_files, data_dir, data_kind=[]):
        # data settings
        self.data_dir = data_dir
        self.SDM_list, self.name_list, self.segmentation_list, self.ori_img_list = self.read_image_segmentation_list(
            txt_files)
        self.data_kind = data_kind
        self.data_kind_no_flip = data_kind
        # check data
        if len(self.SDM_list) != len(self.segmentation_list):
            raise ValueError("The numbers of images and segmentations are different")
        self.length = len(self.SDM_list)

        self.kind_len = len(self.data_kind)

    def __len__(self):
        data_len = self.kind_len * self.length
        return data_len

    def __getitem__(self, id):
        sample = self.get_sample(id)
        return [item for item in sample.values()]

    def get_sample(self, id):
        img_name = self.name_list[id]
        SDM_file_name = self.SDM_list[id]
        seg_file_name = self.segmentation_list[id]
        brain_file_name = self.ori_img_list[id]
        sample = self.load_sample(img_name, SDM_file_name, seg_file_name, brain_file_name)
        return sample

    def load_sample(self, name, SDM_file_name, seg_file_name, brain_file_name):
        """
        Load a segmentation data sample into a dictionary
        """
        # check existing
        if SDM_file_name and not os.path.exists(SDM_file_name):
            raise ValueError(SDM_file_name + ' not exist!')
        if seg_file_name and not os.path.exists(seg_file_name):
            raise ValueError(seg_file_name + ' not exist!')
        if brain_file_name and not os.path.exists(brain_file_name):
            raise ValueError(brain_file_name + ' not exist!')

        sample = {}
        if brain_file_name:
            img = np.load(brain_file_name)
            img = img[np.newaxis, ...]
            print('True')
            img = torch.from_numpy(img)
            # make sure img is 4-dimension
            sample['image'] = img if len(img.shape) <= 4 else img[0, ...]

        if seg_file_name:
            seg = np.load(seg_file_name)
            maskhot_name = seg_file_name.split('/')[-1].replace('mask', 'mask_hot')
            seg_hot_name = os.path.join(seg_file_name.strip(seg_file_name.split('/')[-1]), maskhot_name)
            seg_hot = np.load(seg_hot_name)

            seg = torch.from_numpy(seg)
            seg_hot = torch.from_numpy(seg_hot)
            sample['segmentation'] = seg
            sample['name'] = name
            sample['segmentation_onehot'] = seg_hot if len(seg_hot.shape) <= 4 else seg_hot[0, ...]

        if SDM_file_name:
            sdm = np.load(SDM_file_name)
            sdm = sdm[np.newaxis, ...]
            print('True')
            sdm = torch.from_numpy(sdm)
            sample['sdm'] = sdm if len(sdm.shape) <= 4 else sdm[0, ...]


        return sample

    # get image dir from txt file
    @staticmethod
    def read_image_segmentation_list(text_files):
        """
        read txt file contents and to SDM_list segmentation_list and name_list ang then return them
        """
        SDM_list = []
        segmentation_list = []
        name_list = []
        ori_img_list = []
        segmentation_label_path_ = '/home/amax/disk/tthan/dataset/Fmost_Affine_ants_resample'
        segmentation_label_path_visor = '/home/amax/disk/tthan/dataset/visor_new_resample_affine_'
        if isinstance(text_files, str):
            text_files = [text_files]
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    image_name = line.strip("\n").split('/')[-1].split('.npy')[0]
                    name_list.append(image_name)
                    SDM_list.append(line.strip("\n"))
                    if "Fmost" in image_name:
                        individualinfo_data = image_name.split('_')[-4] + "_norm"
                        segmentation_file_path = segmentation_label_path_ + '/' + individualinfo_data + '/mask.npy'
                        ori_img_file_path = segmentation_label_path_ + '/' + individualinfo_data + '/brain_normalization.npy'
                        segmentation_list.append(segmentation_file_path)
                        ori_img_list.append(ori_img_file_path)
                    else:
                        info = image_name.split('__')[1]
                        info = info.split('_mask_hot')[0]
                        segmentation_file_path = segmentation_label_path_visor + '/' + info + "/" + '/mask.npy'
                        ori_img_file_path = segmentation_label_path_visor + '/' + info + "/" + '/brain_normalization.npy'
                        segmentation_list.append(segmentation_file_path)
                        ori_img_list.append(ori_img_file_path)
        return SDM_list, name_list, segmentation_list, ori_img_list
if __name__ == '__main__':
    print(os.path.realpath("../.."))
