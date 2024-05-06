
import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

n_classes = 9

class RegDataSetMindBoggle(Dataset):
    """
    dataset for image registration
    """

    def __init__(self, txt_files, data_dir, data_kind=[]):
        self.data_dir = data_dir
        self.data_kind = data_kind
        self.image_list, self.name_list, self.segmentation_list, self.ori_img_list = self.read_image_registration_list(
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
        sample = self.load_sample(id_ori, image_name, image_file_name, segmentation_file_name, ori_img_name)
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
        ori_img = np.load(ori_img)
        dd = ori_img.dtype
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
            # dir_test_disp_field='/home/amax/disk/tthan/e'
            # save_name_nii = name + '.nii.gz'
            # savepath_disp_field_niigz = dir_test_disp_field + '/' + save_name_nii
            # sitk.WriteImage(img_save, savepath_disp_field_niigz)

            img = img[np.newaxis, ...]
            ori_img = ori_img[np.newaxis, ...]
            img = torch.from_numpy(img)
            ori_img = torch.from_numpy(ori_img)
            cat_ori_img_img = torch.from_numpy(img_cat)

        if seg_file_name:
            # print ("seg:", seg_file_name)
            seg = np.load(seg_file_name)  # seg intensity from 0 to 16
            if mod == 1:
                seg = np.flip(seg, -1)
            # seg_hot = np.eye(n_classes)[seg].transpose(3, 0, 1, 2)
            # seg_hot = np.load(seg_file_name.replace('mask', 'mask_hot'))
            maskhot_name = seg_file_name.split('/')[-1].replace('mask', 'mask_hot')
            seg_hot_path = os.path.join(seg_file_name.strip(seg_file_name.split('/')[-1]), maskhot_name)
            seg_hot = np.load(seg_hot_path)
            # seg_hot = seg_hot[np.newaxis, ...]
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
        image_file_name = '/home/amax/disk/tthan/SDF/ccf_resample/edt/ccf_resample_CCF_roi_resample_mask_hot.npy'
        ORI_image_file_name = '/home/amax/disk/tthan/dataset/CCF/ccf_resample/CCF_roi_resample/brain_normalization.npy'
        segmentation_file_name = '/home/amax/disk/tthan/dataset/CCF/ccf_resample/CCF_roi_resample/mask.npy' if 'with_seg' in self.data_kind else None
        image_name = image_file_name.split('/')[-1].split('.')[0]
        sample = self.load_sample(0, image_name, image_file_name, segmentation_file_name, ORI_image_file_name)
        return sample

    @staticmethod
    def read_image_registration_list(text_files):
        """
        read txt file contents and to image_list segmentation_list and name_list ang then return them
        """
        image_list = []
        segmentation_list = []
        name_list = []
        ori_img_list = []
        segmentation_path = '/home/amax/disk/tthan/dataset/Fmost/pseudo_by_reg_parameter_affine_real_resample'
        segmentation_path_ = '/home/amax/disk/tthan/dataset/Fmost_Affine_ants_resample'
        if isinstance(text_files, str):
            text_files = [text_files]
        for text_file in text_files:
            with open(text_file) as file:
                for line in file:
                    image_name = line.strip("\n")
                    image_name = image_name.split('/')[-1]
                    name_list.append(image_name)
                    image_list.append(line.strip("\n"))
                    if "Fmost_Affine_ants_resample" not in image_name:
                        individualinfo_data = "Fmost_Affine_ants_" + image_name.split('_')[3] + "_norm_mask_hot"
                        para = image_name.split('_')[7]
                        segmentation_file_path = segmentation_path + '/' + individualinfo_data + '/' + para + '/mask.npy'
                        ori_img_file_path = segmentation_path + '/' + individualinfo_data + '/' + para + '/brain_normalization.npy'
                        segmentation_list.append(segmentation_file_path)
                        ori_img_list.append(ori_img_file_path)
                    else:
                        individualinfo_data = image_name.split('_')[4] + "_norm"
                        segmentation_file_path = segmentation_path_ + '/' + individualinfo_data +  '/mask.npy'
                        ori_img_file_path = segmentation_path_ + '/' + individualinfo_data + '/brain_normalization.npy'
                        segmentation_list.append(segmentation_file_path)
                        ori_img_list.append(ori_img_file_path)
        return image_list, name_list, segmentation_list, ori_img_list

if __name__ == '__main__':
    print(os.path.realpath("../.."))
