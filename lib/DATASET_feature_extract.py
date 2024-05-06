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
                    label_name = 'Fmost_Affine_ants_affine_to_visor_' + info_brain + '_mask_hot.npy'
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
