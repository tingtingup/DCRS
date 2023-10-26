#!/usr/bin/env python
"""
registration exp schedule
last modified by oylei98@163.com on 2021/12/20
"""
import SimpleITK
from torch import squeeze

from .base import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.network_factory import voxel_morph
from lib.network_factory import UNET
from lib.network_factory import unets
from lib.dataset_fmost_indice import Fmost_Dataset
from lib.loss import euclidean,huber_loss
# from lib.datasets import RegDataSetMindBoggle as dataset
# from lib.datasets_oyl import RegDataSetMindBoggle as dataset
# from lib.datasets_fmost_cat import RegDataSetMindBoggle as dataset
import logging
# from lib.datasets_fmost_normaliazation import RegDataSetMindBoggle as dataset_1
# from lib.datasets_fmost_normaliazation import RegDataSetMindBoggle as dataset
from lib.datasets_fmost_generated_validation_GT_edt import RegDataSetMindBoggle as dataset_1
from lib.datasets_fmost_generated_validation_GT_edt import RegDataSetMindBoggle as dataset
# from lib.datasets_fmost_generated_compute_edt import RegDataSetMindBoggle as dataset
import torch.nn.functional as F
import itertools
                                                                     


class Regression_Reg(BaseExperiment):
    def __init__(self, config):
        super(Regression_Reg, self).__init__(config)
        self.weakly_supervised = True if 'with_seg' in self.config['data_kind'] else False
        if self.config['mk_dir']:
            # define exp name and create exp dir
            day = datetime.datetime.now().strftime('%Y_%m_%d_')
            hour = str(int(datetime.datetime.now().strftime('%H')) + 8)
            minute = datetime.datetime.now().strftime('%M')
            self.date_time = day + hour + ':' + minute
            self.exp_name_Regression = 'Regression_{}{}'.format(
                    'unsupervised',
                    '_epoch-{}'.format(self.config['n_epochs']),
                )
            self.exp_name_Registration = 'Registration_{}{}'.format(
                'unsupervised',
                '_epoch-{}'.format(self.config['n_epochs']),
            )
            self.exp_name_Segmentation = 'Segmentation_{}{}'.format(
                'unsupervised',
                '_epoch-{}'.format(self.config['n_epochs']),
            )
            print('exp name:' +'\n', self.exp_name_Regression +'\n', self.exp_name_Registration)
            self.ckpoint_dir_RS = os.path.join(self.config['log_dir'], self.exp_name_Regression, self.date_time)
            self.ckpoint_dir_RT = os.path.join(self.config['log_dir'], self.exp_name_Registration, self.date_time)
            self.ckpoint_dir_RSeg = os.path.join(self.config['log_dir'], self.exp_name_Segmentation, self.date_time)

        else:
            # self.ckpoint_dir_RS = '/home/amax/disk/tthan/wm/code4_unet_dist/MODEL_TTHAN/final_resample/2022_11_08_21:40'
            # # self.ckpoint_dir_RT = '/home/amax/disk/tthan/ori_mini_code_from_oyl_generation/logs_sunrui_generation_edt+ori_oritrain/generation_/Reg_weakly-supervised_epoch-300/2022_11_17_30:55' #### registration
            # self.ckpoint_dir_RT = '/home/amax/disk/tthan/ori_mini_code_from_oyl_generation/compare_experiments_1/yznzheng/generation_/Reg_weakly-supervised_epoch-160/2023_07_30_18:30'
            # self.ckpoint_dir_RSeg = '/home/amax/disk/tthan/ori_mini_code_from_oyl_generation/compare_experiments_1/mindboggle_generation_/Fmost_segmentation/Seg_fully-supervised_epoch-300/2023_05_15_30:39'
            self.ckpoint_dir_RT = self.config['log_dir']
            self.ckpoint_dir_RS = self.config['ckpoint_dir_RS']
            self.ckpoint_dir_RSeg = self.config['ckpoint_dir_RSeg']
            # self.ckpoint_dir = self.config['log_dir'].replace('oyl', 'OYL')
        if not os.path.exists(self.ckpoint_dir_RT):
            os.makedirs(self.ckpoint_dir_RT)
        self.log_dir = os.path.join(self.ckpoint_dir_RT, 'exp_log')
        self.test_flag = self.config['test']

        # self.log_dir = os.path.join(self.config['log_dir'], 'exp_log')

      ##initial regitration and regression model
        self.regression_model = UNET.unet3d(1, 1).cuda()
        self.registration_model = voxel_morph.VoxelMorphCVPR2018(
            input_channel=2, output_channel=3,
            enc_filters=[16, 32, 32, 32, 32], dec_filters=[32, 32, 32, 8, 8]).cuda()
        self.segmentation_model = unets.UNet(in_channel=1, n_classes=self.config['n_classes'], bias=True, BN=True).cuda()
        self.optimizer_regeression_registration = torch.optim.Adam(itertools.chain(self.regression_model.parameters(),self.registration_model.parameters()),lr=self.config['learning_rate'])
        self.optimizer_regeression_segmentation = optim.Adam(
            itertools.chain(self.regression_model.parameters(), self.registration_model.parameters()),
            lr=self.config['learning_rate'])
        # self.optimizer_regeression = optim.Adam(self.regression_model.parameters(), lr=self.config['learning_rate'])
        # self.optimizer_registration = optim.Adam(self.registration_model.parameters(), lr=self.config['learning_rate'])
        # self.optimizer_segmentation = optim.Adam(self.segmentation_model.parameters(), lr=self.config['learning_rate'])
        # self.scheduler_regression = ReduceLROnPlateau(self.optimizer_regeression_registration, 'min', factor=0.5, patience=10,
        #                                               min_lr=1e-5)
        self.scheduler_registration = ReduceLROnPlateau(self.optimizer_regeression_registration, mode='max',
                                                        patience=10,
                                                        factor=0.5, verbose=True, threshold_mode='abs',
                                                        threshold=0.003,
                                                        min_lr=1e-5)
        self.scheduler_segmentation = lr_scheduler.ReduceLROnPlateau(self.optimizer_regeression_segmentation, mode='max',
                                                        patience=10,
                                                        factor=0.5, verbose=True, threshold_mode='abs',
                                                        threshold=0.003,
                                                        min_lr=1e-5)
        self.ckpoint_dir_RS_reuse_path = '/home/amax/disk/tthan/wm/code4_unet_dist/MODEL_TTHAN/final_resample/2022_11_08_21:40'
        resume_model_regression = os.path.join(self.ckpoint_dir_RS_reuse_path, 'epoch-500.pth')
        print('resume_model_regression', resume_model_regression)

        self.initialize_model_(
            self.regression_model, self.regression_model,
            resume_model_regression if os.path.exists(resume_model_regression) else ''
        )
        self.ckpoint_dir_RT_reuse_path = '/home/amax/disk/tthan/ori_mini_code_from_oyl_generation/compare_experiments_1/yznzheng/generation_/Reg_weakly-supervised_epoch-160/2023_07_30_18:30'
        resume_model_registration = os.path.join(self.ckpoint_dir_RT_reuse_path, 'epoch-150_checkpoint.pth')
        # self.ckpoint_dir_RT_reuse_path = '/home/amax/disk/tthan/ori_mini_code_from_oyl_generation/logs_sunrui_generation_edt+ori_oritrain/generation_/Reg_weakly-supervised_epoch-300/2022_11_17_30:55'
        # resume_model_registration = os.path.join(self.ckpoint_dir_RT_reuse_path, 'epoch-300_checkpoint.pth')
        # resume_model = os.path.join(self.ckpoint_dir_RT, 'checkpoint.pth')
             # init/resume model
        finished_epoch_reg, self.reg_best_score = self.initialize_model(
            self.registration_model, self.registration_model,
            resume_model_registration if os.path.exists(resume_model_registration) else ''
        )
        self.ckpoint_dir_RSeg_reuse_path = '/home/amax/disk/tthan/ori_mini_code_from_oyl_generation/compare_experiments_1/mindboggle_generation_/Fmost_segmentation/Seg_fully-supervised_epoch-300/2023_05_15_30:39'
        resume_model_segmentation = os.path.join(self.ckpoint_dir_RSeg_reuse_path, 'epoch-140_checkpoint.pth')
        finished_epoch_seg, self.seg_best_score = self.initialize_model(
            self.segmentation_model, self.segmentation_model,
            resume_model_segmentation if os.path.exists(resume_model_segmentation) else ''
        )
        # # set up loss
        # self.sim_criterions = [('lncc', VoxelMorphLNCC().cuda(), 1)]
        self.sim_criterions = [('mse', nn.MSELoss().cuda(), 1)]
        self.sim_criterions_1 = [('ssim', SSIM3D().cuda(), 1)]
        self.seg_criterions = [
            ('dice', DiceLossMultiClass(n_class=9, weight_type='Uniform', no_bg=False, eps=1e-6).cuda(), 1)]
        self.seg_criterions_ = DiceLossMultiClass(n_class=9, weight_type='Uniform', no_bg=False, softmax=True, eps=1e-6).cuda()
        self.reg_criterions = [('gradient', gradientLoss(norm='L2').cuda(),30)]
        self.regression_criterions = [('Huberloss', nn.HuberLoss().cuda(), 1)]
        # self.sim_regression_criterions = [('euclidean', euclidean(), 1)]


##     initial dataset
        self.fmost_dataset_train = Fmost_Dataset(self.config['img_path'], self.config['label_path'],
                                                 self.config['label_path_'])
        self.fmost_dataset_valadation = Fmost_Dataset(self.config['img_path_valadation'],
                                                      self.config['label_path_validation'],
                                                      self.config['label_path_validation_'])
        self.dataloader_train = DataLoader(self.fmost_dataset_train, batch_size=self.config['batch_size'],
                                           shuffle=False,
                                           num_workers=0)
        self.dataloader_valadation = DataLoader(self.fmost_dataset_valadation, batch_size=self.config['batch_size'],
                                                shuffle=False,
                                                num_workers=0)
        self.fmost_testing_data = Fmost_Dataset(self.config['img_path_test'], self.config['label_path_test'], self.config['label_path_test_'])

        self.testing_data_loader = DataLoader(self.fmost_testing_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0, )
    def setup_log(self):
        if not os.path.isdir(self.ckpoint_dir_RS):
            os.makedirs(self.ckpoint_dir_RS)
        save_dict_to_json(self.config, os.path.join(self.ckpoint_dir_RS, "train_config.json"))
    def train(self):
        self.setup_log()
        self.setup_random_seed()
        # self.setup_model()
        self.logger = self.get_logger(self.log_dir)
        self.current_epoch = int(self.config['begin_epoch']) + 1
        print("Start Training:")
        self.logger.info('starting training!')
        for epoch in range(self.current_epoch, self.config['n_epochs'] + 1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1

    def train_one_epoch(self):
        running_losses = [['Total', 0.0]]
        # running_losses += [[name, 0.0] for name, _, _ in self.regression_criterions]

        running_losses += [[name, 0.0] for name, _, _ in self.sim_criterions]
        running_losses += [[name, 0.0] for name, _, _ in self.sim_criterions_1]
        running_losses += [[name, 0.0] for name, _, _ in self.reg_criterions]
        if self.weakly_supervised:
            running_losses += [[name, 0.0] for name, _, _ in self.seg_criterions]
        # running_losses += [[name, 0.0] for name, _, _ in self.seg_criterions_]
        # steps setting
        start_time = time.time()  # log running time
        batches_per_epoch = len(self.dataloader_train.dataset)
        with trange(batches_per_epoch,
                    desc='Epoch[{}/{}] Train'.format(self.current_epoch, self.config['n_epochs'])) as t:
            train_data_iter = iter(self.dataloader_train)
            self.logger.info('Epoch[{}/{}] Train'.format(self.current_epoch, self.config['n_epochs']))
            for i in t:
                with torch.autograd.set_detect_anomaly(True):
                    target, atlas = next(train_data_iter)
                    img, sdf_label, img_label, name, mask_onehot = target ##source:atlas targed:to registration
                    f_edt_l_atlas, fi ,seg_atlas, seg_hot_atlas = atlas
                    # print(x_img.size())
                    # print(y_label.size())
                    inputs = img.cuda().float()
                    labels = sdf_label.cuda().float()
                    mask_onehot = mask_onehot.cuda().float()

                    self.regression_model.train()
                    self.registration_model.train()
                    self.segmentation_model.train()
                    self.optimizer_regeression_registration.zero_grad()
                    self.optimizer_regeression_segmentation.zero_grad()


                    for p in self.regression_model.parameters():  # reset requires_grad
                        p.requires_grad = True  # they are set to True below in Seger update
                    for p in self.segmentation_model.parameters():
                        p.requires_grad = False
                    for p in self.registration_model.parameters():  # reset requires_grad             -
                        p.requires_grad = True  # they are set to False below in Reger update

                    prd_l = self.regression_model(inputs)
                    # warped_source_image = torch.squeeze(prd_l)
                    # warped_source_image = warped_source_image.detach.numpy()
                    # Img_warped_sorce_indice = sitk.GetImageFromArray(warped_source_image, isVector=False)
                    # # # save_name_nii = 'epoch-{}_batch-{}_warped_edt.nii.gz'.format(self.current_epoch,_)
                    # save_name_nii = name + '_.nii.gz'
                    # savepath_warpdata_niigz = './temp'
                    # sitk.WriteImage(Img_warped_sorce_indice, savepath_warpdata_niigz)

                    f_edt_l_atlas = f_edt_l_atlas.cuda().float()
                    fi_img = fi.cuda().float()
                    disp_field, deform_field = self.registration_model(f_edt_l_atlas, prd_l)
                    w_f_edt = F.grid_sample(f_edt_l_atlas, grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                                            padding_mode='zeros', align_corners=True)
                    warped_target_image_image = F.grid_sample(fi_img,
                                                              grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                                                              ##real
                                                              padding_mode='zeros', align_corners=True)
                    if self.weakly_supervised:
                        warped_source_seg_onehot = F.grid_sample(seg_hot_atlas.cuda().float(),
                                                                 grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                                 mode='bilinear',
                                                                 padding_mode='zeros',
                                                                 align_corners=True)

                    losses = []
                    if self.sim_criterions:
                        losses += [(name, sim_criterion(w_f_edt, prd_l), weight)
                                   for name, sim_criterion, weight in self.sim_criterions]
                    if self.sim_criterions_1:
                        losses += [(name, sim_criterion(warped_target_image_image, inputs), weight)
                               for name, sim_criterion, weight in self.sim_criterions_1]
                    if self.reg_criterions:
                        losses += [(name, reg_criterion(disp_field), weight)
                                   for name, reg_criterion, weight in self.reg_criterions]
                    if self.weakly_supervised:
                        losses += [(name, seg_criterion(warped_source_seg_onehot, mask_onehot), weight)
                                   for name, seg_criterion, weight in self.seg_criterions]


                    self.total_loss = 0
                    for name, subloss, weight in losses:
                        self.total_loss += subloss * weight
                        print('name:',subloss)
                    # update model parameters

                    self.total_loss.backward(retain_graph=True)
                    # self.total_loss.backward()




                    running_losses[0][1] += self.total_loss.item()
                    for k in range(1, len(running_losses)):
                        running_losses[k][1] += losses[k - 1][1].item()

                    for p in self.regression_model.parameters():  # reset requires_grad
                        p.requires_grad = True  # they are set to True below in Seger update
                    for p in self.segmentation_model.parameters():
                        p.requires_grad = True
                    for p in self.registration_model.parameters():  # reset requires_grad             -
                        p.requires_grad = False  # they are set to False below in Reger update
                    seg_output = self.segmentation_model(prd_l)
                    # print ('output:',seg_output.shape)
                    # print('truth:', mask_onehot.shape)
                    # print(output.type())
                    # losseseg = []
                    # losseseg += [(name, seg_criterion(seg_output, mask_onehot), weight)
                    #                for name, seg_criterion, weight in self.seg_criterions_1]
                    # self.total_loss_seg = 0
                    # for name, subloss, weight in losseseg:
                    #     self.total_loss_seg += subloss * weight

                    self.total_segmentation = self.seg_criterions_(seg_output, mask_onehot)
                    print(self.total_segmentation)

                    # self.total_segmentation.backward()
                    self.optimizer_regeression_registration.step()
                    # self.optimizer_regeression_segmentation.step()

                    # running_loss += loss.item()  # average loss over batches
                    duration = time.time() - start_time
                    t.set_postfix_str(
                        '{} lr:{} ({:.3f} sec/batch) {}'.format(
                            ' '.join(['{}_loss: {:.3e}'.format(name, value) for name, value in running_losses]),
                            self.optimizer_regeression_registration.param_groups[0]['lr'],
                            duration,
                            datetime.datetime.now().strftime("%D %H:%M:%S")
                        )
                    )
                    self.logger.info(
                        'loss={}\t lr:{}\t ({:.3f} sec/batch)\t time:{}'.format(self.total_loss,
                                                                                self.optimizer_regeression_registration.param_groups[0]['lr'],
                                                                                duration, datetime.datetime.now().strftime(
                                "%D %H:%M:%S")))

                    t.set_postfix_str('seg_loss: {:.3f}  lr:{} ({:.3f} sec/batch) {}'.format(
                        self.total_segmentation,
                        self.optimizer_regeression_segmentation.param_groups[0]['lr'],
                        duration,
                        datetime.datetime.now().strftime("%D %H:%M:%S")
                    ))
                    # running_loss = 0.0
                    # start_time = time.time()

                    # print statistics
                    # duration = time.time() - start_time


            # aver_loss = self.running_losses[0][1]/batches_per_epoch
            # aver_loss = self.totoal_loss/batches_per_epoch
            # self.scheduler_regression.step(aver_loss)
    def test_iterator(self, mi, ml, m_mask_onehot, target_name):
        with torch.no_grad():
        # regression
            pre_sdf = self.regression_model(mi)
        return pre_sdf
            # pre_sdf = pre_sdf.unsqueeze(1)
            # mi = mi.unsqueeze(1)
        #     mi_presdf_cat = torch.cat((mi, pre_sdf), dim=1)
        #     # f_edt_l = '/home/qulab/disk/tthan/SDF/ccf/edt/CCF_CCF_roi_resample_mask_hot.npy'
        #     # fi = '/home/qulab/disk/tthan/dataset/CCF/CCF_roi_resample/brain.npy'
        #     # fi_mask_onehot = '/home/qulab/disk/tthan/dataset/CCF/CCF_roi_resample/mask_hot.npy'
        #     f_edt_l = '/home/qulab/disk/tthan/SDF/ccf_resample/edt/ccf_resample_CCF_roi_resample_mask_hot.npy'
        #     fi = '/home/qulab/disk/tthan/dataset/CCF/ccf_resample/CCF_roi_resample/brain.npy'
        #     fi_mask_onehot = '/home/qulab/disk/tthan/dataset/CCF/ccf_resample/CCF_roi_resample/mask_hot.npy'
        #
        #     fi_mask_onehot = np.load(fi_mask_onehot)
        #     f_edt_l = np.load(f_edt_l)
        #     fi = np.load(fi)
        #     f_edt_l = f_edt_l[np.newaxis, ...]
        #     fi = fi[np.newaxis, ...]
        #     fi_img_cat = np.concatenate((f_edt_l, fi), 0)
        #     fi_to_r = torch.from_numpy(fi_img_cat)
        #     fi_to_r = fi_to_r.unsqueeze(0)
        #     fi_to_r = fi_to_r.cuda().float()
        #     source_seg_onehot = torch.from_numpy(fi_mask_onehot)
        #     source_seg_onehot = source_seg_onehot.unsqueeze(0)
        #     fi = torch.from_numpy(fi)
        #     # registration
        #     disp_field, deform_field = self.registration_model(mi_presdf_cat, fi_to_r)
        #     warped_source_seg_onehot = F.grid_sample(source_seg_onehot.cuda().float(),
        #                                              grid=deform_field.permute([0, 2, 3, 4, 1]), mode='nearest',
        #                                              padding_mode='zeros',
        #                                              align_corners=True)
        #     if not self.config['test']:
        #         self.test_validation(fi, deform_field, disp_field, warped_source_seg_onehot, fi_to_r, target_name)
        #
        # return warped_source_seg_onehot, disp_field, deform_field, pre_sdf
    def test_validation(self, fi, deform_field,disp_field, warped_source_seg_onehot,fi_to_r,target_name):
        if not self.config['test']:
            dir_name = self.config['data']
            savepath_warpdata = self.config['pseudo_data_dir']
            if not os.path.exists(savepath_warpdata):
                os.makedirs(savepath_warpdata)
            dir_test_warped_image = os.path.join(savepath_warpdata, dir_name)
            if not os.path.exists(dir_test_warped_image):
                os.makedirs(dir_test_warped_image)
            dir_test_warped_edt = os.path.join(dir_test_warped_image, 'warped_edt')
            if not os.path.exists(dir_test_warped_edt):
                os.makedirs(dir_test_warped_edt)
            dir_test_warped_seg = os.path.join(dir_test_warped_image, 'warped_seg')
            if not os.path.exists(dir_test_warped_seg):
                os.makedirs(dir_test_warped_seg)
            dir_test_disp_field = os.path.join(dir_test_warped_image, 'disp_field')
            if not os.path.exists(dir_test_disp_field):
                os.makedirs(dir_test_disp_field)
            dir_test_source_img = os.path.join(dir_test_warped_image, 'source_img_warp')
            if not os.path.exists(dir_test_source_img):
                os.makedirs(dir_test_source_img)
            warped_source_ori_image = F.grid_sample(fi.cuda().float(), grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                                                    padding_mode='zeros', align_corners=True)
            ####
            warped_source_image = torch.squeeze(warped_source_ori_image)
            warped_source_image = warped_source_image.cpu().numpy()
            Img_warped_sorce_edt = sitk.GetImageFromArray(warped_source_image, isVector=False)
            save_name_nii = target_name[0] + '.nii.gz'
            savepath_warpdata_niigz = dir_test_source_img + '/' + save_name_nii
            sitk.WriteImage(Img_warped_sorce_edt, savepath_warpdata_niigz)
            ######
            warped_seg = torch.argmax(warped_source_seg_onehot[0, ...].detach(), 0).cpu().numpy()
            Img_warped_seg = sitk.GetImageFromArray(warped_seg, isVector=False)
            save_name_nii_warped_seg = target_name[0] + '.nii.gz'
            savepath_warpdata_niigz = dir_test_warped_seg + '/' + save_name_nii_warped_seg
            sitk.WriteImage(Img_warped_seg, savepath_warpdata_niigz)
            ####
            disp_field_image = torch.squeeze(disp_field)
            disp_field_image = disp_field_image.cpu().numpy()
            Img_disp_field = sitk.GetImageFromArray(disp_field_image, isVector=False)
            save_name_nii = target_name[0] + '.nii.gz'
            savepath_disp_field_niigz = dir_test_disp_field + '/' + save_name_nii
            sitk.WriteImage(Img_disp_field, savepath_disp_field_niigz)
            #
    def eval(self, dataloader, json_save_dir_reg='', json_save_dir_seg =''):
        euclidean_data = 0.0
        with torch.no_grad():
            self.regression_model.eval()
            self.registration_model.eval()
            self.segmentation_model.eval()
            dice_per_class = torch.zeros(self.config["n_classes"] - 1)  # no background class 0
            running_dice = torch.zeros(self.config["n_classes"] - 1)
            dice_per_class_seg = torch.zeros(self.config["n_classes"] - 1)  # no background class 0
            running_dice_seg = torch.zeros(self.config["n_classes"] - 1)
            # euclidean_data = 0.0
            test_data_iter = iter(dataloader)
            dice = {}
            dice_dict = {}
            dice_seg = {}
            dice_dict_seg = {}
            dice_std = []
            self.jacobian_total = 0
            self.jacobian_total_percentage = 0
            self.jacobian_percentage_std = []
            self.jacobian_std = []


            batches_per_epoch = len(dataloader.dataset)
            total_time = 0
            with trange(batches_per_epoch) as t:
                for o in t:
                    target, atlas = next(test_data_iter)
                    img, sdf_label, img_label, name, mask_onehot = target  ##source:atlas targed:to registration
                    f_edt_l_atlas, fi, seg_atlas, seg_hot_atlas = atlas
                    image_on_device = img.cuda().float()
                    fi_img = fi.cuda().float()
                    image_label_on_device = img_label.cuda().float()
                    f_edt_l_atlas = f_edt_l_atlas.cuda().float()
                    pre_sdf = self.regression_model(image_on_device)

                    if self.current_epoch % 10 == 1 and not self.test_flag:
                        pre_sdf_for_save = torch.squeeze(pre_sdf)
                        pre_sdf_for_save_ = pre_sdf_for_save.cpu().numpy()
                        pre_sdf_for_save_img = sitk.GetImageFromArray(pre_sdf_for_save_, isVector=False)
                        save_name_nii =  'epoch_' + str(self.current_epoch) + str(name) + '.nii.gz'
                        save_path = './temp_new_train_noseg'
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        savepath_warpdata_niigz = './temp_new_train/' + '_' +  save_name_nii
                        sitk.WriteImage(pre_sdf_for_save_img, savepath_warpdata_niigz)

                    disp_field, deform_field = self.registration_model(f_edt_l_atlas, pre_sdf)
                    seg_out = self.segmentation_model(pre_sdf).cpu()

                    # w_f_edt = F.grid_sample(f_edt_l_atlas, grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                    #                         padding_mode='zeros', align_corners=True)
                    warped_source_seg_onehot = F.grid_sample(seg_hot_atlas.cuda().float(),
                                                             grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                             mode='bilinear',
                                                             padding_mode='zeros',
                                                             align_corners=True)
                    warped_target_image_image = F.grid_sample(fi_img,
                                                              grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                              mode='bilinear',
                                                              ##real
                                                              padding_mode='zeros', align_corners=True)

                    if not self.test_flag:
                        self.save_test_result(self, json_save_dir_reg, warped_target_image_image, name, disp_field,
                                         warped_source_seg_onehot, seg_atlas, img_label)

                   #####registration_eval
                    # accumulating dice score in each batch
                    for c in range(1, self.config["n_classes"]):
                        running_dice[c - 1] = metrics.metricEval('dice',
                                                                 torch.argmax(warped_source_seg_onehot[0, ...].detach(),
                                                                              0).cpu().numpy() == c,
                                                                 img_label[0, ...].numpy() == c,
                                                                 num_labels=2)

                        dice_per_class[c - 1] += running_dice[c - 1]
                        dice_dict[c - 1] = float(running_dice[c - 1])
                    # using '100' represent 'avg' for easily sorting
                    dice_dict[100] = float(running_dice.mean())
                    dice[name[0]] = copy.deepcopy(dice_dict)
                    ave_dice = float(running_dice.mean())
                    dice_std.append(ave_dice)
                      #####seg_eval
                    for c in range(1, self.config["n_classes"]):
                        running_dice_seg[c - 1] = metrics.metricEval('dice',
                                                                     torch.max(seg_out.squeeze(), 0)[1].numpy() == c,
                                                                     torch.max(mask_onehot.squeeze(), 0)[1].numpy() == c,
                                                                     num_labels=2)
                        dice_per_class_seg[c - 1] += running_dice_seg[c - 1]
                        dice_dict_seg[c - 1] = float(running_dice_seg[c - 1])

                    dice_dict_seg[100] = float(running_dice_seg.mean())
                    dice_seg[name[0]] = copy.deepcopy(dice_dict_seg)

                    if not self.test_flag:
                        running_dice_test_withouttraning = torch.zeros(self.config["n_classes"] - 1)
                        running_dice_test_withtraning = torch.zeros(self.config["n_classes"] - 1)

                        for c in range(1, self.config["n_classes"]):
                            running_dice_test_withouttraning[c - 1] = metrics.metricEval('dice', seg_atlas[0, ...] == c,
                                                                                         img_label[0, ...].numpy() == c,
                                                                                         num_labels=2)
                        ave_dice_withouttraining = float(running_dice_test_withouttraning.mean())
                        print("withoutraing_{}: Dice Avg: {:.4f} ".format(name, ave_dice_withouttraining) +
                              ' '.join(
                                  ["Dice_{}:{:.3f}".format(self.config["class_name"][c],
                                                           running_dice_test_withouttraning[c]) for c in
                                   range(self.config["n_classes"] - 1)]))

                if not self.test_flag:
                    print("dice_std:", dice_std)
                    std_dice = np.std(dice_std, ddof=1)
                    std_jacobian = np.std(self.jacobian_std, ddof=1)
                    std_jacobian_percentage = np.std(self.jacobian_percentage_std, ddof=1)
                    # GPU_time_average = total_time / batches_per_epoch
                    jacobian_avg = self.jacobian_total / batches_per_epoch
                    jacobian_percentage_avg = self.jacobian_total_percentage / batches_per_epoch


                    # registration_average dice score
                dice_per_class = dice_per_class / batches_per_epoch
                dice_avg = dice_per_class.mean()
                # based on validation size=5, not robust
                if len(dataloader.dataset) > 10:
                    dice['{}'.format('train_avg' if self.config['gen'] else 'test_avg')] = float(dice_avg.numpy())
                else:
                    dice['val_avg'] = float(dice_avg.numpy())
                    if not self.test_flag:
                        dice['jacobian_avg'] = float(jacobian_avg)
                        dice['jacobian_avg_percentage'] = float(jacobian_percentage_avg)
                        # dice['GPU_time_average'] = float(GPU_time_average)
                        dice['std_dice'] = float(std_dice)
                        dice['std_jacobian'] = float(std_jacobian)
                        dice['std_jacobian_percentage'] = float(std_jacobian_percentage)
                if json_save_dir_reg:
                    if os.path.exists(json_save_dir_reg):
                        old_dice = load_jason_to_dict(json_save_dir_reg)
                        old_dice.update(dice)
                        save_dict_to_json(old_dice, json_save_dir_reg)
                    else:
                        save_dict_to_json(dice, json_save_dir_reg)
                    ### segmentation_average_dice_score
                dice_per_class_seg = dice_per_class_seg / batches_per_epoch
                dice_avg_seg = dice_per_class_seg.mean()
                # based on validation size=5, not robust
                if len(dataloader.dataset) > 10:
                    dice_seg['{}'.format('train_avg' if self.config['gen'] else 'test_avg')] = float(dice_avg_seg.numpy())
                else:
                    dice_seg['val_avg'] = float(dice_avg_seg.numpy())
                if json_save_dir_seg:
                    if os.path.exists(json_save_dir_seg):
                        old_dice = load_jason_to_dict(json_save_dir_seg)
                        old_dice.update(dice_seg)
                        save_dict_to_json(old_dice, json_save_dir_seg)
                    else:
                        save_dict_to_json(dice_seg, json_save_dir_seg)

                ###test_save

            return dice_per_class, dice_avg, dice_per_class_seg, dice_avg_seg

    def validate(self):
        start_time = time.time()
        json_save_dir_reg = self.ckpoint_dir_RT + '/' + 'epoch-{}_checkpoint_dice-score.json'.format(self.current_epoch)
        json_save_dir_seg = self.ckpoint_dir_RSeg + '/' + 'epoch-{}_checkpoint_dice-score.json'.format(self.current_epoch)
        # json_save_dir_evalation_dice = self.config['log_dir'] + '/' + 'valadition_epoch-per.json'
        dice_per_class, dice_avg, dice_per_class_seg, dice_avg_seg = self.eval(self.dataloader_valadation, json_save_dir_reg, json_save_dir_seg)
        print('dice_per_class:',dice_per_class)
        print('dice_avg:', dice_avg)
        print('dice_per_class_seg:',dice_per_class_seg)
        print('dice_avg_seg',dice_avg_seg)
        self.scheduler_registration.step(dice_avg)
        self.scheduler_segmentation.step(dice_avg_seg)
        is_best = False
        print("best_score_registration:", self.reg_best_score)
        if dice_avg > self.reg_best_score:
            is_best = True
            self.reg_best_score = dice_avg
        print("best_score_segmentation", self.seg_best_score)
        if dice_avg_seg > self.seg_best_score:
            is_best = True
            self.seg_best_score = dice_avg_seg


        print("Validation_registration: Dice Avg: {:.4f} ".format(dice_avg) +
              ' '.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c], dice_per_class[c]) for c in
                        range(self.config["n_classes"] - 1)]) +
              " {:.3f} sec) {}".format(time.time() - start_time,
                                       datetime.datetime.now().strftime("%D %H:%M:%S")))
        print("Validation_segmentation: Dice Avg: {:.4f} ".format(dice_avg_seg) +
              ' '.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c], dice_per_class_seg[c]) for c in
                        range(self.config["n_classes"] - 1)]) +
              " {:.3f} sec) {}".format(time.time() - start_time,
                                       datetime.datetime.now().strftime("%D %H:%M:%S")))

        # saving registration_model after every setting epochs
        if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
            self.save_checkpoint({'epoch': self.current_epoch,
                                  'model_state_dict': self.registration_model.state_dict(),
                                  'optimizer_state_dict': self.optimizer_regeression_registration.state_dict(),
                                  'best_score': self.reg_best_score},
                                 False, self.ckpoint_dir_RT,
                                 prefix='epoch-{}'.format(self.current_epoch))

        self.save_checkpoint({'epoch': self.current_epoch,
                              'model_state_dict': self.registration_model.state_dict(),
                              'optimizer_state_dict': self.optimizer_regeression_registration.state_dict(),
                              'best_score': self.reg_best_score},
                             is_best, self.ckpoint_dir_RT, )
        # saving segemntation_model after every setting epochs
        if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
            self.save_checkpoint({'epoch': self.current_epoch,
                                  'model_state_dict': self.segmentation_model.state_dict(),
                                  'optimizer_state_dict': self.optimizer_regeression_segmentation.state_dict(),
                                  'best_score': self.seg_best_score},
                                 False, self.ckpoint_dir_RSeg,
                                 prefix='epoch-{}'.format(self.current_epoch))

        self.save_checkpoint({'epoch': self.current_epoch,
                              'model_state_dict': self.segmentation_model.state_dict(),
                              'optimizer_state_dict': self.optimizer_regeression_segmentation.state_dict(),
                              'best_score': self.seg_best_score},
                             is_best, self.ckpoint_dir_RSeg, )
        if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
            self.save_epoch_model = self.ckpoint_dir_RS + '/' + 'epoch-{}'.format(self.current_epoch) + '.pth'
            torch.save(self.regression_model.state_dict(), self.save_epoch_model)



        # self.scheduler_regression.step(euc_avg)
        # self.scheduler_registration.step(dice_avg)
        # is_best = False
        # print("best_score:", self.best_score)
        # if dice_avg > self.best_score:
        #     is_best = True
        #     self.best_score = dice_avg
        # print("Validation: Dice Avg: {:.4f} ".format(dice_avg) + ''.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c], dice_per_class[c]) for c in
        #                 range(self.config["n_classes"] - 1)]) + " {:.3f} sec) {}".format(time.time() - start_time, datetime.datetime.now().strftime("%D %H:%M:%S")))
        # print ("valadation_euc_avg:{:.3f}".format(euc_avg))

        # saving model after every setting epochs
        # if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
        #     self.save_epoch_model_regression = self.ckpoint_dir_RS + '/' + 'epoch-{}'.format(self.current_epoch) + '.pth'
        #     torch.save(self.regression_model.state_dict(), self.save_epoch_model_regression)
        #     self.save_checkpoint({'epoch': self.current_epoch,
        #                           'model_state_dict': self.registration_model.state_dict(),
        #                           'optimizer_state_dict': self.optimizer_registration.state_dict(),
        #                           'best_score': self.best_score},
        #                          False, self.ckpoint_dir_RT,
        #                          prefix='epoch-{}'.format(self.current_epoch))
        #
        # with open(json_save_dir_evalation_dice, 'a') as fr:
        #     fr.write(str(self.current_epoch)+ ':\n')
        #
        #     fr.write("Validation: Dice Avg: {:.4f} ".format(dice_avg) + ''.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c], dice_per_class[c]) for c in
        #                 range(9 - 1)]) + " {:.3f} sec) {}".format(time.time() - start_time, datetime.datetime.now().strftime("%D %H:%M:%S")) +'\n')
        #     fr.write("valadation:{:.3f}".format(euc_avg)+'\n')
    def save_test_result(self, saving_eval_dir, warped_source_ori_image,target_name,disp_field,warped_source_seg_onehot,source_seg,target_seg):
        ##path_makeset
        dir_name = saving_eval_dir.split('/')[-4]
        savepath_warpdata = os.path.join(self.ckpoint_dir_RT, 'test')
        if not os.path.exists(savepath_warpdata):
            os.mkdir(savepath_warpdata)
        dir_test_warped_image = os.path.join(savepath_warpdata, dir_name)
        if not os.path.exists(dir_test_warped_image):
            os.mkdir(dir_test_warped_image)
        dir_test_warped_edt = os.path.join(dir_test_warped_image, 'warped_edt')
        if not os.path.exists(dir_test_warped_edt):
            os.mkdir(dir_test_warped_edt)
        dir_test_warped_seg = os.path.join(dir_test_warped_image, 'warped_seg')
        if not os.path.exists(dir_test_warped_seg):
            os.mkdir(dir_test_warped_seg)
        dir_test_disp_field = os.path.join(dir_test_warped_image, 'disp_field')
        if not os.path.exists(dir_test_disp_field):
            os.mkdir(dir_test_disp_field)
        dir_test_source_img = os.path.join(dir_test_warped_image, 'source_img_warp')
        if not os.path.exists(dir_test_source_img):
            os.mkdir(dir_test_source_img)
        save_name_nii = target_name[0] + '.nii.gz'
        savepath_warp_edt_nii = dir_test_warped_edt + '/' + save_name_nii
        savepath_disp_field_nii = dir_test_disp_field + '/' + save_name_nii
        savepath_warp_seg_nii = dir_test_warped_seg + '/' + save_name_nii
        savepath_warp_source_ori_image = dir_test_source_img + '/' + save_name_nii

      ##save_source_ori_image
        warped_source_image = torch.squeeze(warped_source_ori_image).cpu().numpy()
        Img_warped_sorce_indice = sitk.GetImageFromArray(warped_source_image, isVector=False)
        sitk.WriteImage(Img_warped_sorce_indice, savepath_warp_edt_nii)

        ##save_disp_field
        disp_field_image = torch.squeeze(disp_field).cpu().numpy()
        Img_disp_field = sitk.GetImageFromArray(disp_field_image, isVector=False)
        sitk.WriteImage(Img_disp_field, savepath_disp_field_nii)

        ##save_warped_seg
        warped_seg = torch.argmax(warped_source_seg_onehot[0, ...].detach(), 0).cpu().numpy()
        Img_warped_seg = sitk.GetImageFromArray(warped_seg, isVector=False)
        sitk.WriteImage(Img_warped_seg, savepath_warp_seg_nii)

        ##save_warped_ori_img_source
        warped_source_ori_image_ = torch.squeeze(warped_source_ori_image).detach().cpu().numpy()
        warped_source_ori_image_save = sitk.GetImageFromArray(warped_source_ori_image_, isVector=False)
        sitk.WriteImage(warped_source_ori_image_save, savepath_warp_source_ori_image)

        ###save_test_result

        running_dice_test_withouttraning = torch.zeros(self.config["n_classes"] - 1)
        running_dice_test_withtraning = torch.zeros(self.config["n_classes"] - 1)

        disp_for_jacobian = torch.squeeze(disp_field).permute(1, 2, 3, 0).cpu().numpy()
        jacobian_1 = metrics.jacobian_determinant(disp_for_jacobian)
        jacobian = np.sum(jacobian_1 <= 0)
        voxel_num = disp_for_jacobian.shape[0] * disp_for_jacobian.shape[1] * disp_for_jacobian.shape[2]
        percentage_of_jacbian = jacobian / voxel_num
        self.jacobian_total += jacobian
        self.jacobian_total_percentage += percentage_of_jacbian
        self.jacobian_std.append(jacobian)
        self.jacobian_percentage_std.append(percentage_of_jacbian)
        print("trained_{}: jacobian: {:.4f} ".format(target_name[0], jacobian))
        print("trained_{}: percentage_jacobian: {:.4f} ".format(target_name[0], percentage_of_jacbian))


    def setup_model(self):
        # build registration model
        self.regression_model = UNET.unet3d(1, 1).cuda()
        self.optimizer_regeression = optim.Adam(self.regression_model.parameters(), lr=self.config['learning_rate'])
        self.scheduler_regression = ReduceLROnPlateau(self.optimizer_regeression, 'min', factor=0.5, patience=10,
                                                      min_lr=1e-5)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10, min_lr=1e-5)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=2)

    def test(self):
        print('testing...')
        if self.ckpoint_dir_RT and self.ckpoint_dir_RS:
            # ckpoint_file_RS = os.path.join(self.ckpoint_dir_RS, 'model_best.pth')
            ckpoint_file_RT_test = os.path.join(self.ckpoint_dir_RT, 'epoch-30_checkpoint.pth')
            ckpoint_file_RS_test = os.path.join(self.ckpoint_dir_RS, 'epoch-30_checkpoint.pth')
            ckpoint_dir_RSeg_test = os.path.join(self.ckpoint_dir_RSeg, 'epoch-30_checkpoint.pth')
            print('test_model_regression:',ckpoint_file_RT_test)
            print('test_model_registration:', ckpoint_file_RS_test)
            print('test_model_segmentation:', ckpoint_dir_RSeg_test)

        finished_epoch_reg, self.reg_best_score = self.initialize_model(
            self.registration_model, self.registration_model,
            ckpoint_file_RT_test if os.path.exists(ckpoint_file_RT_test) else ''
        )
        # finished_epoch_reg, best_score = self.initialize_model(self.registration_model, optimizer=None, ckpoint_path=ckpoint_file_RT_test)
        finished_epoch_seg, self.seg_best_score = self.initialize_model(
            self.segmentation_model, self.segmentation_model,
            ckpoint_dir_RSeg_test if os.path.exists(ckpoint_dir_RSeg_test) else ''
        )
        self.initialize_model_(
            self.regression_model, self.regression_model,ckpoint_file_RS_test if os.path.exists(ckpoint_file_RS_test) else ''
        )
        dir_items = ckpoint_file_RT_test.split('/')
        json_save_dir = ''
        for dir_item in dir_items[0:-1]:
            json_save_dir += '/' + dir_item
        json_name = dir_items[-1].replace('.pth', '_dice-score_correct_input.json')
        json_save_dir += '/' + json_name
        dice_per_class, dice_avg, dice_per_class_seg, dice_avg_seg = self.eval(self.testing_data_loader, json_save_dir)
        print('testing dice_registration:', dice_avg.numpy())
        print('dice_per_class_registration:', dice_per_class.numpy())
        print('testing dice_segmentation:', dice_avg_seg.numpy())
        print('dice_per_class_segmentation:', dice_per_class_seg.numpy())

        return float(dice_avg.numpy())

    def generate_data(self):
        print('generating by reg model...')
        pseudo_data_dir = self.config['pseudo_data_dir']
        if not os.path.isdir(pseudo_data_dir):
            os.makedirs(pseudo_data_dir)

        # cleaning old json file
        for dir_name in os.listdir(pseudo_data_dir):
            if dir_name.endswith('json'):
                os.remove(os.path.join(pseudo_data_dir, dir_name))

        generate_data = dataset(self.config['augmenting_list_file'],
                                self.config['data_dir'],
                                # ['with_seg', 'gt'],
                                ['indice', 'cat', 'with_seg', 'gt'],
                                )
        self.generate_data_loader = DataLoader(generate_data,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               )

        # initialize model
        self.setup_model()
        if '.pth' not in self.ckpoint_dir:
            if self.config['gen_from_best']:
                ckpoint_file = os.path.join(self.ckpoint_dir, 'epoch-60_checkpoint.pth')
            else:
                ckpoint_file = os.path.join(self.ckpoint_dir, 'checkpoint.pth')
        else:
            ckpoint_file = self.ckpoint_dir
        last_epoch, best_score = self.initialize_model(self.model, optimizer=None, ckpoint_path=ckpoint_file)

        with torch.no_grad():
            self.model.eval()
            dice_per_class = torch.zeros(self.config["n_classes"] - 1)
            running_dice = torch.zeros(self.config["n_classes"] - 1)
            dice = {}
            dice_per_img = {}
            self.generate_data_iter = iter(self.generate_data_loader)
            for j in range(len(self.generate_data_loader)):
                # (source_image, source_image_brain, source_seg, name_source, source_seg_onehot) = source_data
                # (target_image, targed_image_brain, target_seg, target_name, target_seg_onehot) = target_data
                ((source_image, source_image_brain, source_seg, source_name, source_seg_onehot),
                 (target_image, target_image_brain, target_seg, target_name, target_seg_onehot)) = next(self.generate_data_iter)
                source_image_on_device = source_image.cuda().float()
                target_image_on_device = target_image.cuda().float()
                # parameter = [1, 0.2, 0.4, 0.6, 0.7]
                parameter = [0.5, 0.3]
                for para in parameter:
                    disp_field, deform_field = self.model(source_image_on_device,
                                                                           target_image_on_device,para)

                # grid_sample function: input should [N, C, H, W, D], grid should [N, H, W, D, 3]
                # warped_source_seg_onehot = F.grid_sample(source_seg_onehot.cuda().float(),
                #                                          grid=deform_field.permute([0, 2, 3, 4, 1]),
                #                                          mode='bilinear',
                #                                          padding_mode='zeros',
                #                                          align_corners=True)
                #
                # warped_source_image = F.grid_sample(source_image_brain.cuda().float(),
                #                                         grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                #                                         padding_mode='zeros', align_corners=True)
                    warped_source_seg_onehot = F.grid_sample(target_seg_onehot.cuda().float(),
                                                         grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                         mode='bilinear',
                                                         padding_mode='zeros',
                                                         align_corners=True)

                    warped_source_image = F.grid_sample(target_image_brain.cuda().float(),
                                                    grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                                                    padding_mode='zeros', align_corners=True)

                # TODO: may not reasonable: uint8 ??
                    warped_source_seg = torch.argmax(warped_source_seg_onehot.squeeze(), 0).cpu().numpy()
                # warped_source_seg = np.uint8(warped_source_seg)
                    seg_hot = np.eye(self.config['n_classes'])[warped_source_seg].transpose(3, 0, 1, 2)
                # # seg_hot = np.uint8(seg_hot)
                # seg_hot_flip = np.flip(seg_hot, -1)
                # seg_hot_flip = np.uint8(seg_hot_flip)

                    warped_source_image = warped_source_image.squeeze().cpu().numpy()
                # warped_source_seg_flip = np.flip(warped_source_seg, -1)
                # warped_source_image_flip = np.flip(warped_source_image, -1)

                # -----------------------------pseudo data generate----------------------------------
                    pseudo_data_dir_save_path = '/home/qulab/disk/tthan/dataset/Fmost/pseudo_by_reg_parameter'

                    saving_dir = os.path.join(pseudo_data_dir_save_path, target_name[0])
                    print(saving_dir)
                    print('saving {}... '.format(saving_dir))
                    if not os.path.isdir(saving_dir):
                        os.makedirs(saving_dir, 0o777)


                    saving_dir = os.path.join(saving_dir, str(para))
                    if not os.path.isdir(saving_dir):
                        os.makedirs(saving_dir, 0o777)
                    # saving deform field
                    disp_field = disp_field.squeeze().cpu().numpy()
                    np.save(saving_dir + '/field.npy', disp_field)

                    # weakly_data_path = '/home/qulab/disk/tthan/dataset/Fmost_weakly'
                    # # ---------------------------saving weakly and flip data-------------------------------
                    # saving_dir = os.path.join(weakly_data_path, target_name[0])
                    # if not os.path.isdir(saving_dir):
                    #     os.makedirs(saving_dir, 0o777)
                    np.save(saving_dir + '/mask.npy', warped_source_seg)
                    # np.save(saving_dir + '/mask_flip.npy', warped_source_seg_flip)
                    np.save(saving_dir + '/mask_hot.npy', seg_hot)
                    # np.save(saving_dir + '/mask_flip_hot.npy', seg_hot_flip)
                    # saving in nii.gz format for visualize
                    mask = sitk.GetImageFromArray(warped_source_seg)
                    # mask_flip = sitk.GetImageFromArray(warped_source_seg_flip)
                    sitk.WriteImage(mask, saving_dir + '/weakly_mask.nii.gz')
                    # sitk.WriteImage(mask_flip, saving_dir + '/weakly_mask_flip.nii.gz')
                    # augmentation_data_path = '/home/qulab/disk/tthan/dataset/Fmost_augmentation'
                    # # ---------------------------saving augmentation and flip data-------------------------------
                    # saving_dir = os.path.join(augmentation_data_path, target_name[0])
                    # if not os.path.isdir(saving_dir):
                    #     os.makedirs(saving_dir, 0o777)
                    np.save(saving_dir + '/brain.npy', warped_source_image)
                    # np.save(saving_dir + '/brain_flip.npy', warped_source_image_flip)
                    # saving in nii.gz format for visualize
                    image = sitk.GetImageFromArray(warped_source_image)
                    # image_flip = sitk.GetImageFromArray(warped_source_image_flip)
                    sitk.WriteImage(image, saving_dir + '/weakly_brain.nii.gz')
                    # sitk.WriteImage(image_flip, saving_dir + '/weakly_brain_flip.nii.gz')

                    # calculate and saviguochguochag data dice
                    for c in range(1, self.config["n_classes"]):
                        running_dice[c - 1] = metrics.metricEval('dice',
                                                                 torch.argmax(
                                                                     warped_source_seg_onehot[0, ...],
                                                                     0).cpu().numpy() == c,
                                                                 target_seg[0, ...].numpy() == c,
                                                                 num_labels=2)
                        dice_per_class[c - 1] += running_dice[c - 1]
                        dice_per_img[c - 1] = float(running_dice[c - 1].numpy())
                    dice_per_img[100] = float(running_dice.mean().numpy())
                    dice[target_name[0]] = copy.deepcopy(dice_per_img)
                    dice_per_class = dice_per_class / (j + 1)
                    dice_avg = dice_per_class.mean()
                    dice['train_avg'] = float(dice_avg.numpy())
                    save_dict_to_json(dice, os.path.join(pseudo_data_dir, 'model-{}_training_dice.json'.format(last_epoch)))
    def get_logger(self, filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger



'''
    def generate_data(self):
        print('generating by reg model...')
        pseudo_data_dir = self.config['pseudo_data_dir']
        if not os.path.isdir(pseudo_data_dir):
            os.makedirs(pseudo_data_dir)

        # cleaning old json file
        for dir_name in os.listdir(pseudo_data_dir):
            if dir_name.endswith('json'):
                os.remove(os.path.join(pseudo_data_dir, dir_name))

        generate_data = dataset(self.config['training_list_file'],
                                self.config['data_dir'],
                                ['with_seg', 'gt'],
                                )
        self.generate_data_loader = DataLoader(generate_data,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               )

        # initialize model
        self.setup_model()
        if '.pth' not in self.ckpoint_dir:
            if self.config['gen_from_best']:
                ckpoint_file = os.path.join(self.ckpoint_dir, 'model_best.pth')
            else:
                ckpoint_file = os.path.join(self.ckpoint_dir, 'checkpoint.pth')
        else:
            ckpoint_file = self.ckpoint_dir
        last_epoch, best_score = self.initialize_model(self.model, optimizer=None, ckpoint_path=ckpoint_file)

        with torch.no_grad():
            self.model.eval()
            dice_per_class = torch.zeros(self.config["n_classes"] - 1)
            running_dice = torch.zeros(self.config["n_classes"] - 1)
            dice = {}
            dice_per_img = {}
            self.generate_data_iter = iter(self.generate_data_loader)
            for j in range(len(self.generate_data_loader)):
                ((source_image, source_seg, source_name, source_seg_onehot),
                 (target_image, target_seg, target_name, target_seg_onehot)) = next(self.generate_data_iter)
                source_image_on_device = source_image.cuda()
                target_image_on_device = target_image.cuda()
                disp_field, warped_source_image, deform_field = self.model(source_image_on_device,
                                                                                       target_image_on_device)

                # grid_sample function: input should [N, C, H, W, D], grid should [N, H, W, D, 3]
                warped_source_seg_onehot = F.grid_sample(source_seg_onehot.cuda().float(),
                                                         grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                         mode='bilinear',
                                                         padding_mode='zeros',
                                                         align_corners=True)

                # TODO: may not reasonable: uint8 ??
                warped_source_seg = torch.argmax(warped_source_seg_onehot.squeeze(), 0).cpu().numpy()
                warped_source_seg = np.uint8(warped_source_seg)
                seg_hot = np.eye(self.config['n_classes'])[warped_source_seg].transpose(3, 0, 1, 2)
                seg_hot = np.uint8(seg_hot)
                seg_hot_flip = np.flip(seg_hot, -1)
                seg_hot_flip = np.uint8(seg_hot_flip)

                warped_source_image = warped_source_image.squeeze().cpu().numpy()
                warped_source_seg_flip = np.flip(warped_source_seg, -1)
                warped_source_image_flip = np.flip(warped_source_image, -1)

                # -----------------------------pseudo data generate----------------------------------
                saving_dir = os.path.join(pseudo_data_dir, target_name[0])
                print('saving {}... '.format(saving_dir))
                if not os.path.isdir(saving_dir):
                    os.makedirs(saving_dir, 0o777)

                # saving deform field
                disp_field = disp_field.squeeze().cpu().numpy()
                np.save(saving_dir + '/field.npy', disp_field)

                # ---------------------------saving weakly and flip data-------------------------------
                saving_dir = os.path.join('../data/weakly_data', target_name[0])
                if not os.path.isdir(saving_dir):
                    os.makedirs(saving_dir, 0o777)
                np.save(saving_dir + '/mask.npy', warped_source_seg)
                np.save(saving_dir + '/mask_flip.npy', warped_source_seg_flip)
                np.save(saving_dir + '/mask_hot.npy', seg_hot)
                np.save(saving_dir + '/mask_flip_hot.npy', seg_hot_flip)
                # saving in nii.gz format for visualize
                mask = sitk.GetImageFromArray(warped_source_seg)
                mask_flip = sitk.GetImageFromArray(warped_source_seg_flip)
                sitk.WriteImage(mask, saving_dir + '/weakly_mask.nii.gz')
                sitk.WriteImage(mask_flip, saving_dir + '/weakly_mask_flip.nii.gz')

                # ---------------------------saving augmentation and flip data-------------------------------
                saving_dir = os.path.join('../data/augmentation_data', target_name[0])
                if not os.path.isdir(saving_dir):
                    os.makedirs(saving_dir, 0o777)
                np.save(saving_dir + '/brain.npy', warped_source_image)
                np.save(saving_dir + '/brain_flip.npy', warped_source_image_flip)
                # saving in nii.gz format for visualize
                image = sitk.GetImageFromArray(warped_source_image)
                image_flip = sitk.GetImageFromArray(warped_source_image_flip)
                sitk.WriteImage(image, saving_dir + '/weakly_brain.nii.gz')
                sitk.WriteImage(image_flip, saving_dir + '/weakly_brain_flip.nii.gz')

                # calculate and saviguochguochag data dice
                for c in range(1, self.config["n_classes"]):
                    running_dice[c - 1] = metrics.metricEval('dice',
                                                             torch.argmax(
                                                                 warped_source_seg_onehot[0, ...],
                                                                 0).cpu().numpy() == c,
                                                             target_seg[0, ...].numpy() == c,
                                                             num_labels=2)
                    dice_per_class[c - 1] += running_dice[c - 1]
                    dice_per_img[c - 1] = float(running_dice[c - 1].numpy())
                dice_per_img[100] = float(running_dice.mean().numpy())
                dice[target_name[0]] = copy.deepcopy(dice_per_img)
                dice_per_class = dice_per_class / (j + 1)
                dice_avg = dice_per_class.mean()
                dice['train_avg'] = float(dice_avg.numpy())
                save_dict_to_json(dice, os.path.join(pseudo_data_dir, 'model-{}_training_dice.json'.format(last_epoch)))
'''

#

savepath_warpdata1 = "/home/qulab/disk/tthan/ori_mini_code_from_oyl/warp_img_test_mask_oyl"
savepath_warpdata_source = "/home/qulab/disk/tthan/ori_mini_code_from_oyl/warp_img_test_mask_oyl_source"


# pp = warped_source_seg_onehot[0, ...]
# ss = pp.shape
# oo = pp[1,:,:,:]
# tt = torch.unique(oo)

# accumulating dice score in each batch
