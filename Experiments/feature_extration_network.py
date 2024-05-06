import time
from datetime import datetime
# from base import *
import numpy as np
import torch
import UNET
from torch import optim
from lib.dataloader import Fmost_Dataset
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss_G import dist_loss, diceLoss1, euclidean, dist_loss_mul, color_loss, huber_loss, MAE, RMSE
import logging
import SimpleITK as sitk
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from param_dict import save_dict_to_json, load_jason_to_dict
import datetime

class G():
    def __init__(self, config):
        self.config = config
        if self.config['mk_dir']:
            # define exp name and create exp dir
            day = datetime.datetime.now().strftime('%Y_%m_%d_')
            hour = str(int(datetime.datetime.now().strftime('%H')) + 8)
            minute = datetime.datetime.now().strftime('%M')
            self.date_time = day + hour + ':' + minute
            self.ckpoint_dir = os.path.join(self.config['log_dir'], self.date_time)
        else:
            self.ckpoint_dir = self.config['log_dir']
        self.lr = self.config['learning_rate']
        self.log_dir = os.path.join(self.ckpoint_dir, 'exp_log')


    def setup_train_data(self):
        self.fmost_dataset_train = Fmost_Dataset(self.config['train_dataset'])
        self.fmost_dataset_valadation = Fmost_Dataset(self.config['validation_dataset'])
        self.dataloader_train = DataLoader(self.fmost_dataset_train, batch_size=self.config['batch_size'], shuffle=False,
                                           num_workers=0)
        self.dataloader_valadation = DataLoader(self.fmost_dataset_valadation, batch_size=self.config['batch_size'], shuffle=False,
                                                num_workers=0)
    def setup_test_data(self):
        self.fmost_testing_data = Fmost_Dataset(self.config['test_dataset'])
        self.testing_data_loader = DataLoader(self.fmost_testing_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0, )
    def train(self):
        self.setup_log()
        self.setup_random_seed()
        self.setup_model()
        self.setup_train_data()
        self.logger = self.get_logger(self.log_dir)
        if self.config['resume_from_best']:
            resume_model = os.path.join(self.ckpoint_dir, 'epoch-300.pth')
        else:
            resume_model = os.path.join(self.ckpoint_dir, 'checkpoint.pth')
        self.current_epoch = int(self.config['begin_epoch']) + 1
        self.logger.info('starting training!')
        for epoch in range(self.current_epoch, self.config['n_epochs'] + 1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
        self.logger.info('Finished Training: {}'.format(self.ckpoint_dir))

    def train_one_epoch(self):
        batches_per_epoch = len(self.dataloader_train.dataset)
        dataloader_train_iter = iter(self.dataloader_train)
        total_loss = 0
        self.logger.info('Epoch[{}/{}] Train'.format(self.current_epoch, self.config['n_epochs']))
        for i in range(batches_per_epoch):
            start_time = time.time()
            x_img, y_label = next(dataloader_train_iter)
            inputs = x_img.cuda().float()
            labels = y_label.cuda().float()
            self.model.train()
            self.optimizer.zero_grad()
            predi_x = self.model(inputs)
            loss = huber_loss(labels, predi_x)
            lo = 0
            lo += loss
            lo = lo.item()
            total_loss += lo
            loss.backward()
            self.optimizer.step()
            duration = time.time() - start_time
            self.logger.info('loss={}\t lr:{}\t ({:.3f} sec/batch)\t time:{}'.format(lo, self.optimizer.param_groups[0]['lr'],                                                                         duration,datetime.datetime.now().strftime("%D %H:%M:%S")))
        aver_loss = total_loss / batches_per_epoch
        self.scheduler.step(aver_loss)

    def validate(self):
        start_time = time.time()
        json_save_dir = self.ckpoint_dir + '/' + 'valadition_enu_epoch-per.json'
        euc_avg = self.eval(self.dataloader_valadation, json_save_dir)
        print("epoch:{}\tValidation: euc_avg: {:.4f} ".format(self.current_epoch, euc_avg) +
              " {:.3f} sec) {}".format(time.time() - start_time, datetime.datetime.now().strftime("%D %H:%M:%S")))
        self.save_epoch_model = self.ckpoint_dir + '/' + 'epoch-{}'.format(self.current_epoch) + '.pth'
        if self.current_epoch % 10 == 0:
            torch.save(self.model.state_dict(), self.save_epoch_model)

    def eval(self, dataloader, saving_eval_dir=''):
        euclidean_data = 0.0
        with torch.no_grad():
            self.model.eval()
            test_data_iter = iter(dataloader)
            batches_per_epoch = len(dataloader.dataset)
            for _ in range(batches_per_epoch):
                x_img_test, y_label_test = next(test_data_iter)
                x_img_test = x_img_test.cuda().float()
                y_label_test = y_label_test.cuda().float()
                predi_x_test = self.model(x_img_test)
                predi_x_test = torch.squeeze(predi_x_test)
                y_label_test = torch.squeeze(y_label_test)
                predi_x_test = predi_x_test.detach().cpu().numpy()
                y_label_test = y_label_test.detach().cpu().numpy()

                dir_name = 'generated_SDM'
                savepath_warpdata = "../result"
                if not os.path.exists(savepath_warpdata):
                    os.makedirs(savepath_warpdata)
                dir_test_warped_image = os.path.join(savepath_warpdata, dir_name)
                if not os.path.exists(dir_test_warped_image):
                    os.makedirs(dir_test_warped_image)
                Img_generate = sitk.GetImageFromArray(predi_x_test, isVector=False)
                save_name_nii = 'genenrated_sdm.nii.gz'
                generat_sdf_save_path_niigz = os.path.join(dir_test_warped_image, save_name_nii)
                sitk.WriteImage(Img_generate, generat_sdf_save_path_niigz)
                # simlarity = RMSE(y_label_test, predi_x_test)
                # simlarity = euclidean(y_label_test, predi_x_test)
                # euclidean_data += simlarity

            # euc_avg = euclidean_data/len(dataloader.dataset)
            # print("euc_avg:", euc_avg)
            # if saving_eval_dir:
            #     with open(saving_eval_dir, 'a') as fr:
            #             info_save = 'epoch={}\t enc={}\n'.format(self.current_epoch, euc_avg)
        #     #             fr.write(info_save)
        # return euc_avg
    def test(self):
        # self.logger.info('testing...')
        self.setup_model()
        if '.pth' not in self.ckpoint_dir:
            ckpoint_file = os.path.join(self.ckpoint_dir, 'epoch-500.pth')
        else:
            ckpoint_file = self.ckpoint_dir
        self.initialize_model(self.model, optimizer=None, ckpoint_path=ckpoint_file)
        self.setup_test_data()
        # define json save dir for saving validation score
        dir_items = ckpoint_file.split('/')
        json_save_dir = ''
        for dir_item in dir_items[0:-1]:
            json_save_dir += dir_item + '/'
        json_name = dir_items[-1].replace('.pth', '_dice-score_correct_input.json')
        json_save_dir += '/' + json_name
        euc_avg = self.eval(self.testing_data_loader, json_save_dir)
        # self.logger.info('testing dice:', euc_avg)
        # return float(euc_avg)

    def setup_random_seed(self):
        torch.manual_seed(self.config['random_seed'])
        torch.cuda.manual_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        random.seed(self.config['random_seed'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def setup_log(self):
        if not os.path.isdir(self.ckpoint_dir):
            os.makedirs(self.ckpoint_dir)
        save_dict_to_json(self.config, os.path.join(self.ckpoint_dir, "train_config.json"))

    def setup_model(self):
        # build registration model
        self.model = UNET.unet3d(1, 1).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10, min_lr=1e-5)

    @staticmethod
    def initialize_model(model, optimizer=None, ckpoint_path=None):
        """
        Initilaize a feature_extract_model with saved checkpoins, or random values
        :param model: a pytorch feature_extract_model to be initialized
        :param optimizer: optional, optimizer whose parameters can be restored from saved checkpoints
        :param ckpoint_path: The path of saved checkpoint
        :return: currect epoch and best validation score
        """
        # initialize model by existed parameters
        if ckpoint_path:
            if os.path.isfile(ckpoint_path):
                print("=> loading checkpoint '{}'".format(ckpoint_path))

                model.load_state_dict(torch.load(ckpoint_path, map_location=next(
                    model.parameters()).device))
                model = model.cuda()
                print("=> loaded checkpoint '{}' (epoch)".format(ckpoint_path))
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(ckpoint_path))
        else:
            model.weights_init()

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

#
# if __name__ == '__main__':
#     train_model()
