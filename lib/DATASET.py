import SimpleITK as sitk
import torchvision.transforms as tfs
import numpy as np
import torch
import os
from PIL import Image
from scipy.ndimage import zoom
from torch.utils import data
import cv2
from scipy import ndimage

class Dataset(data.Dataset):
    def __init__(self, datapath_img, datapath_label, datapath_label_, datapath_img_visor, datapath_label_visor, datapath_label_visor_):
        img_visor_append = []
        self.imgs = []
        self.img_visor_append = []

        with open(datapath_img, 'r') as fr:
            lines = fr.readlines()
            for key, line in enumerate(lines):
                # print("key:",key)
                # print("line:",line)
                fixed = line.split()[0]
                moving = line.split()[1]

                self.read_fix(fixed, datapath_label, datapath_label_)
                moving = moving.strip()
                self.read_moving(moving, datapath_label_visor, datapath_label_visor_)

        image_pair = []

        for i in range(len(self.imgs)):
            singel_pair = {}
            singel_pair['fixed'] = self.imgs[i]
            singel_pair['moving'] = self.img_visor_append[i]
            image_pair.append(singel_pair)

        self.image_pair = image_pair

    def read_fix(self,fixed_path, datapath_label,datapath_label_):
        img = fixed_path.strip()
        info_brain = fixed_path.split('/')
        name = info_brain[-3] + '_' + info_brain[-2]
        if 'pseudo_by_reg_parameter' in info_brain[-4]:
            # label_name = 'Fmost_' + info_brain[-3] + "_" + info_brain[-2] + '_mask_hot.npy'
            label_name = info_brain[-3] + "_" + info_brain[-2] + '_mask_hot.npy'
            # print("label", label_name)
            label = os.path.join(datapath_label, label_name)
            mask_label = img.replace('brain_normalization', 'mask')
            mask_onehot = img.replace('brain_normalization', 'mask_hot')
            self.imgs.append([img, label, mask_label, name, mask_onehot])
        else:
            info_brain = info_brain[-2]
            label_name = 'Fmost_Affine_ants_resample_' + info_brain + '_mask_hot.npy'
            # label_name = 'Fmost_Affine_ants_' + info_brain + '_mask_hot.npy'
            label = os.path.join(datapath_label_, label_name)
            mask_label = img.replace('brain_normalization', 'mask')
            mask_onehot = img.replace('brain_normalization', 'mask_hot')
            self.imgs.append([img, label, mask_label, name, mask_onehot])

    def read_moving(self, moving_path, datapath_label_visor, datapath_label_visor_):
        img_visor = moving_path.strip()
        info_brain = moving_path.split('/')
        name_visor = info_brain[-2]
        if 'pseudo_by_reg_parameter_affine' in info_brain[-4]:
            label_name = info_brain[-3] + "_" + info_brain[-2] + '_mask_hot.npy'
            # print("label", label_name)
            label_visor = os.path.join(datapath_label_visor, label_name)
            mask_label_visor = img_visor.replace('brain_normalization', 'mask')
            mask_onehot_visor = img_visor.replace('brain_normalization', 'mask_hot')
            self.img_visor_append.append([img_visor, label_visor, mask_label_visor, name_visor, mask_onehot_visor])
        else:
            info_brain = info_brain[-2]
            label_name ='visor_new_resample_affine__' + info_brain + '_mask_hot.npy'  ###resample
            label_visor = os.path.join(datapath_label_visor_, label_name)
            mask_label_visor = img_visor.replace('brain_normalization', 'mask')
            mask_onehot_visor = img_visor.replace('brain_normalization', 'mask_hot')
            self.img_visor_append.append([img_visor, label_visor, mask_label_visor, name_visor, mask_onehot_visor])

    def __getitem__(self, index):
        x_path, y_path, mask, name, mask_onehot = self.image_pair[index]['fixed']
        f_edt_l_path, fi_path, fi_seg_path, fi_name_visor, fi_seg_hot_path = self.image_pair[index]['moving']

        # x_path, y_path, mask, name, mask_onehot = self.image_pair[index]

        img_x = np.load(x_path)
        img_y = np.load(y_path)
        mask_ = np.load(mask)
        mask_onehot = np.load(mask_onehot)
        img_x = img_x[np.newaxis, ...]
        img_y = img_y[np.newaxis, ...]
        mask_ = mask_[np.newaxis, ...]
        x_transform = torch.from_numpy(img_x)
        y_transform = torch.from_numpy(img_y)
        mask_transform = torch.from_numpy(mask_)
        mask_hot_transform = torch.from_numpy(mask_onehot)

        ####visor
        # f_edt_l_path, fi_path, fi_seg_path, fi_name_visor, fi_seg_hot_path = self.imgs_visor[index]
        f_edt_l_ = np.load(f_edt_l_path)
        f_edt_l_ = f_edt_l_[np.newaxis, ...]
        f_edt_l_1 = torch.from_numpy(f_edt_l_)

        fi_numpy = np.load(fi_path)
        fi_numpy_ = fi_numpy[np.newaxis, ...]
        fi_numpy_1 = torch.from_numpy(fi_numpy_)

        seg = np.load(fi_seg_path)  # seg intensity from 0 to 16
        print('fix_hot:',fi_seg_hot_path)
        seg_hot = np.load(fi_seg_hot_path)
        seg = seg[np.newaxis, ...]
        seg = torch.from_numpy(seg)
        seg_hot = torch.from_numpy(seg_hot)

        return [x_transform, y_transform, mask_transform, name, mask_hot_transform], [f_edt_l_1, fi_numpy_1, seg, fi_name_visor, seg_hot]

    def __len__(self):
        return len(self.imgs)

# if __name__ == '__main__':
#     image = cv2.imread('/home/yasin//test.png')
#     img_new = process_image(image, 224)
#     cv2.imshow("img_new", img_new)
#     cv2.waitKey()



