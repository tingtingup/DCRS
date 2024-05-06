#!/usr/bin/env python

import SimpleITK
from torch import squeeze

from .base import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.network_factory import voxel_morph
from lib.network_factory import UNET
from lib.network_factory import unets
from lib.dataloader import Dataset

import logging
import torch.nn.functional as F
import itertools


class Regression_Reg_Seg(BaseExperiment):
    def __init__(self, config):
        super(Regression_Reg_Seg, self).__init__(config)
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
            self.exp_name_Regression_visor = 'Regression_visor{}{}'.format(
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
            self.ckpoint_dir_RS_visor = os.path.join(self.config['log_dir'], self.exp_name_Regression_visor, self.date_time)
            self.ckpoint_dir_RT = os.path.join(self.config['log_dir'], self.exp_name_Registration, self.date_time)
            self.ckpoint_dir_RSeg = os.path.join(self.config['log_dir'], self.exp_name_Segmentation, self.date_time)

        else:
            self.ckpoint_dir_RT = self.config['log_dir']
            self.ckpoint_dir_RS = self.config['ckpoint_dir_RS']
            self.ckpoint_dir_RS_visor = self.config['ckpoint_dir_RS_visor']
            self.ckpoint_dir_RSeg = self.config['ckpoint_dir_RSeg']
            # self.ckpoint_dir = self.config['log_dir'].replace('oyl', 'OYL')
        if not os.path.exists(self.ckpoint_dir_RT):
            os.makedirs(self.ckpoint_dir_RT)
        self.log_dir = os.path.join(self.ckpoint_dir_RT, 'exp_log')
        self.test_flag = self.config['test']
        print('test_flag:',self.test_flag)

        # self.log_dir = os.path.join(self.config['log_dir'], 'exp_log')

      ##initial regitration and regression model
        self.feature_extract = UNET.unet3d(1, 1).cuda()
        self.feature_extract_visor = UNET.unet3d(1, 1).cuda()
        self.registration_model = voxel_morph.VoxelMorphCVPR2018(
            input_channel=2, output_channel=3,
            enc_filters=[16, 32, 32, 32, 32], dec_filters=[32, 32, 32, 8, 8]).cuda()
        self.segmentation_model = unets.UNet(in_channel=1, n_classes=self.config['n_classes'], bias=True, BN=True).cuda()
        self.optimizer_feature_extract = optim.Adam(self.feature_extract.parameters(), lr=self.config['learning_rate'])
        self.optimizer_feature_extract_visor = optim.Adam(self.feature_extract_visor.parameters(), lr=self.config['learning_rate'])
        self.optimizer_registration = optim.Adam(self.registration_model.parameters(), lr=self.config['learning_rate'])
        self.optimizer_segmentation = optim.Adam(self.segmentation_model.parameters(), lr=self.config['learning_rate'])
        self.scheduler_regression = ReduceLROnPlateau(self.optimizer_feature_extract, 'max', factor=0.5, patience=10,
                                                      min_lr=1e-6)
        self.scheduler_regression_visor = ReduceLROnPlateau(self.optimizer_feature_extract_visor, 'max', factor=0.5, patience=10,
                                                      min_lr=1e-6)

        self.scheduler_registration = ReduceLROnPlateau(self.optimizer_registration, mode='max',
                                                        patience=10,
                                                        factor=0.5, verbose=True, threshold_mode='abs',
                                                        threshold=0.003,
                                                        min_lr=1e-6)
        self.scheduler_segmentation = lr_scheduler.ReduceLROnPlateau(self.optimizer_segmentation,
                                                                     mode='max',
                                                                     patience=10,
                                                                     factor=0.5, verbose=True, threshold_mode='abs',
                                                                     threshold=0.003,
                                                                     min_lr=1e-6)
        self.ckpoint_dir_RS_reuse_path = '../model/pretrain'
        resume_model_regression = os.path.join(self.ckpoint_dir_RS_reuse_path, 'regression.pth')

        self.initialize_model_(
            self.feature_extract, self.feature_extract,
            resume_model_regression if os.path.exists(resume_model_regression) else ''
        )

        self.ckpoint_dir_RT_reuse_path = '../model/pretrain'
        resume_model_registration = os.path.join(self.ckpoint_dir_RT_reuse_path, 'registration.pth')
             # init/resume model
        finished_epoch_reg, self.reg_best_score = self.initialize_model(
            self.registration_model, self.registration_model,
            resume_model_registration if os.path.exists(resume_model_registration) else ''
        )

        self.ckpoint_dir_RSeg_reuse_path = '../model/pretrain'
        resume_model_segmentation = os.path.join(self.ckpoint_dir_RSeg_reuse_path, 'segmentation.pth')
        finished_epoch_seg, self.seg_best_score = self.initialize_model(
            self.segmentation_model, self.segmentation_model,
            resume_model_segmentation if os.path.exists(resume_model_segmentation) else ''
        )
        # # set up loss
        # self.sim_criterions = [('lncc', VoxelMorphLNCC().cuda(), 1)]
        self.sim_criterions = [('mse', nn.MSELoss().cuda(), 1)]
        self.seg_criterions = [
            ('dice', DiceLossMultiClass(n_class=9, weight_type='Uniform', no_bg=False, eps=1e-6).cuda(), 1)]
        self.seg_criterions_modality_fixed =[('dice_fmost', DiceLossMultiClass(n_class=9, weight_type='Uniform', no_bg=False, softmax=True, eps=1e-6).cuda(),1)]
        self.seg_criterions_modality_moving = [('dice_visor',
                                     DiceLossMultiClass(n_class=9, weight_type='Uniform', no_bg=False, softmax=True,
                                                        eps=1e-6).cuda(), 1)]
        self.reg_criterions = [('gradient', gradientLoss(norm='L2').cuda(),30)]

##     initial dataset
        self.Dataset_train = Dataset(self.config['training_list_file'])
        self.Dataset_valadation = Dataset(self.config['validation_list_file'])
        self.dataloader_train = DataLoader(self.Dataset_train, batch_size=self.config['batch_size'],
                                           shuffle=False,
                                           num_workers=0)
        self.dataloader_valadation = DataLoader(self.Dataset_valadation, batch_size=self.config['batch_size'],
                                                shuffle=False,
                                                num_workers=0)
        self.Dataset_test = Dataset(self.config['testing_list_file'])
        self.dataloader_test = DataLoader(self.Dataset_test,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)

    def setup_log(self):
        if not os.path.isdir(self.ckpoint_dir_RS):
            os.makedirs(self.ckpoint_dir_RS)
        if not os.path.isdir(self.ckpoint_dir_RS_visor):
            os.makedirs(self.ckpoint_dir_RS_visor)
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
            self.train_one_epoch()       # self.train_one_epoch()       # self.train_one_epoch()
            self.validate()
            self.current_epoch += 1

    def train_one_epoch(self):
        running_losses = [['Total', 0.0]]
        running_losses += [[name, 0.0] for name, _, _ in self.sim_criterions]
        running_losses += [[name, 0.0] for name, _, _ in self.reg_criterions]
        if self.weakly_supervised:
            running_losses += [[name, 0.0] for name, _, _ in self.seg_criterions]
        running_losses += [[name, 0.0] for name, _, _ in self.seg_criterions_modality_fixed]
        running_losses += [[name, 0.0] for name, _, _ in self.seg_criterions_modality_moving]
        
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
                    f_edt_l_atlas, fi, seg_atlas, fi_name_visor, seg_hot_atlas = atlas

                    inputs = img.cuda().float()
                    labels = sdf_label.cuda().float()
                    mask_onehot = mask_onehot.cuda().float()
                    input_visor = f_edt_l_atlas.cuda().float()
                    seg_hot_atlas = seg_hot_atlas.cuda().float()
                    self.feature_extract.train()
                    self.registration_model.train()
                    self.segmentation_model.train()
                    self.optimizer_feature_extract.zero_grad()
                    self.optimizer_segmentation.zero_grad()
                    self.optimizer_registration.zero_grad()
                    prd_l = self.feature_extract(inputs)
                    f_edt_l_atlas_ = self.feature_extract(input_visor)
                    # prd_l = torch.exp(-prd_l)
                    # f_edt_l_atlas_ = torch.exp(-f_edt_l_atlas_)
                    fi_img = fi.cuda().float()
                    disp_field, deform_field = self.registration_model(f_edt_l_atlas_, prd_l)
                    w_f_edt = F.grid_sample(f_edt_l_atlas_, grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                                            padding_mode='zeros', align_corners=True)
                    # warped_target_image_image = F.grid_sample(input_visor,
                    #                                           grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                    #                                           ##real
                    #                                           padding_mode='zeros', align_corners=True)
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
                    if self.reg_criterions:
                        losses += [(name, reg_criterion(disp_field), weight)
                                   for name, reg_criterion, weight in self.reg_criterions]
                    if self.weakly_supervised:
                        losses += [(name, seg_criterion(warped_source_seg_onehot, mask_onehot), weight)
                                   for name, seg_criterion, weight in self.seg_criterions]

                    seg_output = self.segmentation_model(prd_l)
                    seg_output_1 = self.segmentation_model(f_edt_l_atlas_)

                    if self.seg_criterions_modality_fixed:
                        losses += [(name, seg_criterions_modality_fixed(seg_output, mask_onehot), weight)
                                   for name, seg_criterions_modality_fixed, weight in self.seg_criterions_modality_fixed]
                    if self.seg_criterions_modality_moving:
                        losses += [(name, seg_criterions_modality_moving(seg_output_1, seg_hot_atlas), weight)
                                   for name, seg_criterions_modality_moving, weight in self.seg_criterions_modality_moving]

                    self.total_loss = 0
                    for name, subloss, weight in losses:
                        self.total_loss += subloss * weight
                        print('name:', subloss)
                    self.total_loss.backward()

                    running_losses[0][1] += self.total_loss.item()
                    for k in range(1, len(running_losses)):
                        running_losses[k][1] += losses[k - 1][1].item()


                    self.optimizer_feature_extract.step()
                    # self.optimizer_feature_extract_visor.step()
                    self.optimizer_segmentation.step()
                    self.optimizer_registration.step()

                    # running_loss += loss.item()  # average loss over batches
                    duration = time.time() - start_time
                    t.set_postfix_str(
                        '{} registration_lr:{} segmentation_lr:{} feature_extract_lr:{} ({:.3f} sec/batch) {}'.format(
                            ' '.join(['{}_loss: {:.3e}'.format(name, value) for name, value in running_losses]),
                            self.optimizer_registration.param_groups[0]['lr'],
                            self.optimizer_segmentation.param_groups[0]['lr'],
                            self.optimizer_feature_extract.param_groups[0]['lr'],
                            duration,
                            datetime.datetime.now().strftime("%D %H:%M:%S")
                        )
                    )
                    self.logger.info(
                        'epoch:{}\t loss={}\t lr_registration:{}\t lr_segmentation:{}\t lr_regeression:{}\t ({:.3f} sec/batch)\t time:{}'.format(self.current_epoch, self.total_loss,
                                                                                self.optimizer_registration.param_groups[0]['lr'],
                                                                                self.optimizer_segmentation.param_groups[0]['lr'],
                                                                                self.optimizer_feature_extract.param_groups[0]['lr'],
                                                                                duration, datetime.datetime.now().strftime(
                                "%D %H:%M:%S")))



    def eval(self, dataloader, json_save_dir_reg='', json_save_dir_seg ='',json_save_dir_seg_fixed =''):
        euclidean_data = 0.0
        with torch.no_grad():
            self.feature_extract.eval()
            # self.feature_extract_visor.eval()
            self.registration_model.eval()
            self.segmentation_model.eval()
            dice_per_class = torch.zeros(self.config["n_classes"] - 1)  # no background class 0
            running_dice = torch.zeros(self.config["n_classes"] - 1)
            dice_per_class_seg = torch.zeros(self.config["n_classes"] - 1)  # no background class 0
            running_dice_seg = torch.zeros(self.config["n_classes"] - 1)
            dice_per_class_seg_fixed = torch.zeros(self.config["n_classes"] - 1)  # no background class 0
            running_dice_seg_visor = torch.zeros(self.config["n_classes"] - 1)
            # euclidean_data = 0.0
            test_data_iter = iter(dataloader)
            dice = {}
            dice_dict = {}
            dice_seg = {}
            dice_seg_visor = {}
            dice_dict_seg = {}
            dice_dict_seg_visor = {}
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
                    f_edt_l_atlas, fi, seg_atlas, fi_name_visor, seg_hot_atlas = atlas
                    image_on_device = img.cuda().float()
                    fi_img = fi.cuda().float()
                    image_label_on_device = img_label.cuda().float()
                    f_edt_l_atlas = f_edt_l_atlas.cuda().float()
                    pre_sdf = self.feature_extract(image_on_device)
                    f_edt_l_atlas_ = self.feature_extract(f_edt_l_atlas)

                    # pre_sdf = torch.exp(-pre_sdf)
                    # f_edt_l_atlas_ = torch.exp(-f_edt_l_atlas_)

                    disp_field, deform_field = self.registration_model(f_edt_l_atlas_, pre_sdf)
                    seg_out = self.segmentation_model(pre_sdf).cpu()
                    seg_out_ = self.segmentation_model(f_edt_l_atlas_).cpu()

                    # w_f_edt = F.grid_sample(f_edt_l_atlas, grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                    #                         padding_mode='zeros', align_corners=True)
                    warped_source_seg_onehot = F.grid_sample(seg_hot_atlas.cuda().float(),
                                                             grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                             mode='bilinear',
                                                             padding_mode='zeros',
                                                             align_corners=True)
                    warped_target_image_image = F.grid_sample(f_edt_l_atlas,
                                                              grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                              mode='bilinear',
                                                              ##real
                                                              padding_mode='zeros', align_corners=True)

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
                      #####fmost_seg_eval
                    for c in range(1, self.config["n_classes"]):
                        running_dice_seg[c - 1] = metrics.metricEval('dice',
                                                                     torch.max(seg_out.squeeze(), 0)[1].numpy() == c,
                                                                     torch.max(mask_onehot.squeeze(), 0)[1].numpy() == c,
                                                                     num_labels=2)
                        dice_per_class_seg[c - 1] += running_dice_seg[c - 1]
                        dice_dict_seg[c - 1] = float(running_dice_seg[c - 1])

                    dice_dict_seg[100] = float(running_dice_seg.mean())
                    dice_seg[name[0]] = copy.deepcopy(dice_dict_seg)
                    ###visorsg_eval

                    for c in range(1, self.config["n_classes"]):
                        running_dice_seg_visor[c - 1] = metrics.metricEval('dice',
                                                                     torch.max(seg_out_.squeeze(), 0)[1].numpy() == c,
                                                                     torch.max(seg_hot_atlas.squeeze(), 0)[1].numpy() == c,
                                                                     num_labels=2)
                        dice_per_class_seg_fixed[c - 1] += running_dice_seg_visor[c - 1]
                        dice_dict_seg_visor[c - 1] = float(running_dice_seg_visor[c - 1])

                    dice_dict_seg_visor[100] = float(running_dice_seg_visor.mean())
                    dice_seg_visor[name[0]] = copy.deepcopy(dice_dict_seg_visor)

                    # registration_average dice score
                dice_per_class = dice_per_class / batches_per_epoch
                dice_avg = dice_per_class.mean()
                # based on validation size=5, not robust
                if len(dataloader.dataset) > 10:
                    dice['{}'.format('train_avg' if self.config['gen'] else 'test_avg')] = float(dice_avg.numpy())
                else:
                    dice['val_avg'] = float(dice_avg.numpy())
                if json_save_dir_reg:
                    if os.path.exists(json_save_dir_reg):
                        old_dice = load_jason_to_dict(json_save_dir_reg)
                        old_dice.update(dice)
                        save_dict_to_json(old_dice, json_save_dir_reg)
                    else:
                        save_dict_to_json(dice, json_save_dir_reg)
                    ### fmost_segmentation_average_dice_score
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
                ### visor_segmentation_average_dice_score
                dice_per_class_seg_fixed = dice_per_class_seg_fixed / batches_per_epoch
                dice_avg_seg_fixed = dice_per_class_seg_fixed.mean()
                # based on validation size=5, not robust
                if len(dataloader.dataset) > 10:
                    dice_seg_visor['{}'.format('train_avg' if self.config['gen'] else 'test_avg')] = float(
                        dice_avg_seg_fixed.numpy())
                else:
                    dice_seg_visor['val_avg'] = float(dice_avg_seg_fixed.numpy())
                if json_save_dir_seg_fixed:
                    if os.path.exists(json_save_dir_seg_fixed):
                        old_dice = load_jason_to_dict(json_save_dir_seg_fixed)
                        old_dice.update(dice_seg)
                        save_dict_to_json(old_dice, json_save_dir_seg_fixed)
                    else:
                        save_dict_to_json(dice_seg, json_save_dir_seg_fixed)


                ###test_save

            return dice_per_class, dice_avg, dice_per_class_seg, dice_avg_seg,dice_per_class_seg_fixed, dice_avg_seg_fixed

    def validate(self):
        start_time = time.time()
        json_save_dir_reg = self.ckpoint_dir_RT + '/' + 'epoch-{}.json'.format(self.current_epoch)
        json_save_dir_seg = self.ckpoint_dir_RSeg + '/' + 'epoch-{}'.format(self.current_epoch)
        json_save_dir_seg_fixed = self.ckpoint_dir_RSeg + '/' + 'epoch-{}'.format(
            self.current_epoch)
        # json_save_dir_evalation_dice = self.config['log_dir'] + '/' + 'valadition_epoch-per.json'
        dice_per_class, dice_avg, dice_per_class_seg, dice_avg_seg, dice_per_class_seg_fixed, dice_avg_seg_fixed = self.eval(self.dataloader_valadation, json_save_dir_reg, json_save_dir_seg, json_save_dir_seg_fixed)

        self.scheduler_registration.step(dice_avg)
        self.scheduler_segmentation.step(dice_avg_seg_fixed)
        is_best = False
        print("best_score_registration:", self.reg_best_score)
        if dice_avg > self.reg_best_score:
            is_best = True
            self.reg_best_score = dice_avg
        print("best_score_segmentation", self.seg_best_score)
        # if dice_avg_seg > self.seg_best_score:
        #     is_best = True
        #     self.seg_best_score = dice_avg_seg
        if dice_avg_seg_fixed > self.seg_best_score:
            is_best = True
            self.seg_best_score = dice_avg_seg_fixed


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
        print("Validation_segmentation_visor: Dice Avg: {:.4f} ".format(dice_avg_seg_fixed) +
              ' '.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c], dice_per_class_seg_fixed[c]) for c in
                        range(self.config["n_classes"] - 1)]) +
              " {:.3f} sec) {}".format(time.time() - start_time,
                                       datetime.datetime.now().strftime("%D %H:%M:%S")))

        # saving registration_model after every setting epochs
        if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
            self.save_checkpoint({'epoch': self.current_epoch,
                                  'model_state_dict': self.registration_model.state_dict(),
                                  'optimizer_state_dict': self.optimizer_registration.state_dict(),
                                  'best_score': self.reg_best_score},
                                 False, self.ckpoint_dir_RT,
                                 prefix='epoch-{}'.format(self.current_epoch))

        self.save_checkpoint({'epoch': self.current_epoch,
                              'model_state_dict': self.registration_model.state_dict(),
                              'optimizer_state_dict': self.optimizer_registration.state_dict(),
                              'best_score': self.reg_best_score},
                             is_best, self.ckpoint_dir_RT, )
        # saving segemntation_model after every setting epochs
        if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
            self.save_checkpoint({'epoch': self.current_epoch,
                                  'model_state_dict': self.segmentation_model.state_dict(),
                                  'optimizer_state_dict': self.optimizer_segmentation.state_dict(),
                                  'best_score': self.seg_best_score},
                                 False, self.ckpoint_dir_RSeg,
                                 prefix='epoch-{}'.format(self.current_epoch))

        self.save_checkpoint({'epoch': self.current_epoch,
                              'model_state_dict': self.segmentation_model.state_dict(),
                              'optimizer_state_dict': self.optimizer_segmentation.state_dict(),
                              'best_score': self.seg_best_score},
                             is_best, self.ckpoint_dir_RSeg, )
        if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
            self.save_epoch_model = self.ckpoint_dir_RS + '/' + 'epoch-{}'.format(self.current_epoch) + '.pth'
            torch.save(self.feature_extract.state_dict(), self.save_epoch_model)

        if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
            self.save_epoch_model = self.ckpoint_dir_RS_visor + '/' + 'epoch-{}'.format(self.current_epoch) + '.pth'
            torch.save(self.feature_extract_visor.state_dict(), self.save_epoch_model)


    def setup_model(self):
        # build registration model
        self.feature_extract = UNET.unet3d(1, 1).cuda()
        self.optimizer_feature_extract = optim.Adam(self.feature_extract.parameters(), lr=self.config['learning_rate'])
        self.scheduler_regression = ReduceLROnPlateau(self.optimizer_feature_extract, 'min', factor=0.5, patience=10,
                                                      min_lr=1e-5)

    def test(self):
        print('testing...')
        if self.ckpoint_dir_RT and self.ckpoint_dir_RS:

            ckpoint_file_RT_test = os.path.join(self.ckpoint_dir_RT, 'registration.pth')
            ckpoint_file_RS_test = os.path.join(self.ckpoint_dir_RS, 'regression.pth')
            ckpoint_dir_RSeg_test = os.path.join(self.ckpoint_dir_RSeg, 'segmentation.pth')


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
            self.feature_extract, self.feature_extract,ckpoint_file_RS_test if os.path.exists(ckpoint_file_RS_test) else ''
        )

        dir_items = ckpoint_file_RT_test.split('/')
        json_save_dir = ''
        for dir_item in dir_items[0:-1]:
            json_save_dir += '/' + dir_item
        json_name = dir_items[-1].replace('.pth', '_dice-score_correct_input.json')
        json_save_dir += '/' + json_name
        dice_per_class, dice_avg, dice_per_class_seg, dice_avg_seg = self.eval(self.dataloader_test, json_save_dir)
        return float(dice_avg.numpy())


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

