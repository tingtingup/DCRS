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
n_classes = 9
class Dataset(data.Dataset):
    def __init__(self, datapath_img):
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
                self.read_fix(fixed)
                moving = moving.strip()
                self.read_moving(moving)
        image_pair = []
        for i in range(len(self.imgs)):
            singel_pair = {}
            singel_pair['fixed'] = self.imgs[i]
            singel_pair['moving'] = self.img_visor_append[i]
            image_pair.append(singel_pair)

        self.image_pair = image_pair

    def read_fix(self, fixed_path):
        datapath_label = '/home/amax/disk/tthan/SDF/Fmost_generation_resample/edt'
        datapath_label_ = '/home/amax/disk/tthan/SDF/fmost_ants_affine_resample/edt'
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

    def read_moving(self, moving_path):
        datapath_label_visor = '/home/amax/disk/tthan/SDF/visor_new_resample_affine_generation/edt'
        datapath_label_visor_ = '/home/amax/disk/tthan/SDF/visor_new_resample_affine/edt'
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
        # img_y = np.exp(-img_y)
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
        # fi_numpy = np.exp(-fi_numpy)
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

    def load_sample(self, name, SDM_file_name, seg_file_name, brain_file_name, sdm_file_name):
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

        if sdm_file_name:
            sdm = np.load(sdm_file_name)
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

class Fmost_Dataset(data.Dataset):
    def __init__(self, datapath_img):
        imgs = []
        datapath_label = '/home/amax/disk/tthan/SDF/visor_new_resample_affine/edt'
        datapath_label_ = '/home/amax/disk/tthan/SDF/visor_new_resample_affine/edt'
        with open(datapath_img, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                img = line.strip()
                info_brain = line.split('/')
                if 'pseudo_by_reg_parameter' in info_brain[-4]:
                    label_name = info_brain[-3] + "_" + info_brain[-2] + '_mask_hot.npy'
                    print("label", label_name)
                    label = os.path.join(datapath_label, label_name)
                    imgs.append([img, label])
                else:
                    info_brain = info_brain[-2]
                    # label_name = 'Fmost_Affine_ants_affine_to_visor_' + info_brain + '_mask_hot.npy'
                    label_name = 'visor_new_resample_affine__' + info_brain + '_mask_hot.npy'
                    label = os.path.join(datapath_label_, label_name)
                    imgs.append([img, label])
        self.imgs = imgs


    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = np.load(x_path)
        img_y = np.load(y_path)
        img_x = img_x[np.newaxis, ...]
        x_transform = torch.from_numpy(img_x)
        y_transform = torch.from_numpy(img_y)
        return x_transform, y_transform


    def __len__(self):
        return len(self.imgs)
# if __name__ == '__main__':
#     image = cv2.imread('/home/yasin//test.png')
#     img_new = process_image(image, 224)
#     cv2.imshow("img_new", img_new)
#     cv2.waitKey()



