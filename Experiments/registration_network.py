import logging
from lib.dataloader import Dataset
from base import *
from lib.network_factory import voxel_morph

class RegistrationExperiment(BaseExperiment):
    def __init__(self, config):
        super(RegistrationExperiment, self).__init__(config)
        self.weakly_supervised = True if 'with_seg' in self.config['data_kind'] else False
        if self.config['mk_dir']:
            day = datetime.datetime.now().strftime('%Y_%m_%d_')
            hour = str(int(datetime.datetime.now().strftime('%H')) + 8)
            minute = datetime.datetime.now().strftime('%M')
            self.date_time = day + hour + ':' + minute
            print(self.config['data_kind'])
            if 'with_seg' in self.config['data_kind']:
                self.exp_name = 'Reg_{}{}'.format(
                    'weakly-supervised',
                    '_epoch-{}'.format(self.config['n_epochs']),
                )
            else:
                print('not in withseg')
                self.exp_name = 'Reg_{}{}'.format(
                    'unsupervised',
                    '_epoch-{}'.format(self.config['n_epochs']),
                )
            print('exp name:', self.exp_name)
            self.ckpoint_dir = os.path.join(self.config['log_dir'], self.exp_name, self.date_time)
        else:
            self.ckpoint_dir = self.config['log_dir']
        self.log_dir = os.path.join(self.ckpoint_dir, 'exp_log')

    def setup_log(self):
        if not os.path.isdir(self.ckpoint_dir):
            os.makedirs(self.ckpoint_dir)
        save_dict_to_json(self.config, os.path.join(self.ckpoint_dir, "train_config.json"))

    def setup_model(self):
        # build registration model
        self.model = voxel_morph.VoxelMorphCVPR2018(
            input_channel=2, output_channel=3,
            enc_filters=[16, 32, 32, 32, 32], dec_filters=[32, 32, 32, 8, 8]).cuda()
        self.sim_criterions = [('mse', nn.MSELoss().cuda(), 1)]
        self.seg_criterions = [
            ('dice', DiceLossMultiClass(n_class=1, weight_type='Uniform', no_bg=False, eps=1e-6).cuda(), 1)]
        self.reg_criterions = [('gradient', gradientLoss(norm='L2').cuda(), 1)]
        # set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                        patience=10,
                                                        factor=0.5, verbose=True, threshold_mode='abs',
                                                        threshold=0.003,
                                                        min_lr=1e-5)

    def setup_train_data(self):
        print("Initializing dataloader")
        # set up data loader
        self.Dataset_train = Dataset(self.config['training_list_file'])
        self.Dataset_valadation = Dataset(self.config['validation_list_file'])
        self.dataloader_train = DataLoader(self.Dataset_train, batch_size=self.config['batch_size'],
                                           shuffle=False,
                                           num_workers=0)
        self.dataloader_valadation = DataLoader(self.Dataset_valadation, batch_size=self.config['batch_size'],
                                                shuffle=False,
                                                num_workers=0)


    def train(self):
        self.setup_train()
        self.setup_train_data()
        print("Training {}".format(self.ckpoint_dir))
        self.logger = self.get_logger(self.log_dir)
        if not self.config['resume_from_best']:
            resume_model = os.path.join(self.ckpoint_dir, 'checkpoint.pth')
        else:
            resume_model = os.path.join(self.ckpoint_dir, 'model_best.pth')
        finished_epoch, self.best_score = self.initialize_model(
            self.model, self.optimizer,
            resume_model if os.path.exists(resume_model) else ''
        )
        self.current_epoch = int(self.config['begin_epoch']) + 1
        print("Start Training:")
        self.logger.info('starting training!')
        for epoch in range(self.current_epoch, self.config['n_epochs'] + 1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
        self.logger.info('Finished Training: {}'.format(self.ckpoint_dir))
        print('Finished Training: {}'.format(self.ckpoint_dir))

    def train_one_epoch(self):
        running_losses = [['Total', 0.0]]
        running_losses += [[name, 0.0] for name, _, _ in self.sim_criterions]
        if self.weakly_supervised:
            running_losses += [[name, 0.0] for name, _, _ in self.seg_criterions]
        running_losses += [[name, 0.0] for name, _, _ in self.reg_criterions]
        start_time = time.time()  # log running time
        batches_per_epoch = len(self.dataloader_train.dataset)
        with trange(batches_per_epoch,
                    desc='Epoch[{}/{}] Train'.format(self.current_epoch, self.config['n_epochs'])) as t:
            train_data_iter = iter(self.dataloader_train)
            self.logger.info('Epoch[{}/{}] Train'.format(self.current_epoch, self.config['n_epochs']))
            for i in t:
                source_data, target_data = next(train_data_iter)  ##source:atlas targed:to registration
                if self.weakly_supervised:
                    # (source_image, source_image_brain, source_seg, name_source, source_seg_onehot) = source_data
                    # (target_image, targed_image_brain, target_seg, target_name, target_seg_onehot) = target_data
                    (targed_image_brain, target_image, target_seg, target_name, target_seg_onehot) = source_data
                    (source_image_brain, source_image, source_seg, name_source, source_seg_onehot) = target_data  ##source:atlas targed:to registration
                source_image_brain_device = source_image.cuda().float()
                target_image_brain_device = target_image.cuda().float()
                self.model.train()
                self.optimizer.zero_grad()
                disp_field, deform_field = self.model(source_image_brain_device, target_image_brain_device)
                warped_source_image = F.grid_sample(source_image_brain_device,
                                                    grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear', padding_mode='zeros', align_corners=True) ##real

                if self.weakly_supervised:
                    warped_source_seg_onehot = F.grid_sample(source_seg_onehot.cuda().float(),
                                                             grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                             mode='bilinear',
                                                             padding_mode='zeros',
                                                             align_corners=True)
                losses = []
                if self.sim_criterions:
                    losses += [(name, sim_criterion(warped_source_image, target_image_brain_device), weight)
                               for name, sim_criterion, weight in self.sim_criterions]  ##yyy
                if self.weakly_supervised:
                    losses += [(name, seg_criterion(warped_source_seg_onehot, target_seg_onehot.cuda().float()), weight)
                               for name, seg_criterion, weight in self.seg_criterions]

                # displacement field regularization loss
                losses += [(name, reg_criterion(disp_field), weight)
                           for name, reg_criterion, weight in self.reg_criterions]

                total_loss = 0
                for name, subloss, weight in losses:
                    total_loss += subloss * weight
                # update model parameters
                total_loss.backward()
                self.optimizer.step()
                # print loss info initialization
                running_losses[0][1] += total_loss.item()
                for k in range(1, len(running_losses)):
                    running_losses[k][1] += losses[k - 1][1].item()
                # print statistics
                duration = time.time() - start_time
                t.set_postfix_str(
                    '{} lr:{} ({:.3f} sec/batch) {}'.format(
                        ' '.join(['{}_loss: {:.3e}'.format(name, value) for name, value in running_losses]),
                        self.optimizer.param_groups[0]['lr'],
                        duration,
                        datetime.datetime.now().strftime("%D %H:%M:%S")
                    )
                )
                self.logger.info(
                    'loss={}\t lr:{}\t ({:.3f} sec/batch)\t time:{}'.format(total_loss,
                                                                            self.optimizer.param_groups[0]['lr'],
                                                                            duration, datetime.datetime.now().strftime(
                            "%D %H:%M:%S")))

                # variable reset
                for k in range(len(running_losses)):
                    running_losses[k][1] = 0
                start_time = time.time()

    def eval(self, dataloader, saving_eval_dir=''):
        with torch.no_grad():
            self.model.eval()
            dice_per_class = torch.zeros(self.config["n_classes"] - 1)  # no background class 0
            running_dice = torch.zeros(self.config["n_classes"] - 1)
            test_data_iter = iter(dataloader)
            dice = {}
            dice_dict = {}
            batches_per_epoch = len(dataloader.dataset)
            total_time = 0
            with trange(batches_per_epoch) as t:
                for _ in t:
                    start_time = time.time()
                    ((target_image, target_ori_image, target_seg, target_name, target_seg_onehot),  # source_image:atlas
                     (source_image, source_ori_image, source_seg, source_name, source_seg_onehot)) = next(test_data_iter)
                    # source_image_device = source_image.cuda().float()
                    # target_image_device = target_image.cuda().float()
                    source_image_device = source_ori_image.cuda().float()
                    target_image_device = target_ori_image.cuda().float()
                    disp_field, deform_field = self.model(source_image_device, target_image_device)
                    warped_target_ori_image = F.grid_sample(source_ori_image.cuda().float(),
                                                            grid=deform_field.permute([0, 2, 3, 4, 1]), mode='bilinear',
                                                            padding_mode='zeros', align_corners=True)
                    duration = time.time() - start_time
                    total_time += duration
                    warped_source_seg_onehot = F.grid_sample(source_seg_onehot.cuda().float(),
                                                             grid=deform_field.permute([0, 2, 3, 4, 1]),
                                                             mode='nearest',
                                                             padding_mode='zeros',
                                                             align_corners=True)
                    # accumulating dice score in each batch
                    for c in range(1, self.config["n_classes"]):
                        running_dice[c - 1] = metrics.metricEval('dice',
                                                                 torch.argmax(warped_source_seg_onehot[0, ...].detach(),
                                                                              0).cpu().numpy() == c,
                                                                 target_seg[0, ...].numpy() == c,
                                                                 num_labels=2)

                        dice_per_class[c - 1] += running_dice[c - 1]
                        dice_dict[c - 1] = float(running_dice[c - 1])
                    # using '100' represent 'avg' for easily sorting
                    dice_dict[100] = float(running_dice.mean())
                    dice[target_name[0]] = copy.deepcopy(dice_dict)

            # average dice score
            dice_per_class = dice_per_class / batches_per_epoch
            dice_avg = dice_per_class.mean()
            # based on validation size=5, not robust
            if len(dataloader.dataset) > 40:
                dice['{}'.format('train_avg' if self.config['gen'] else 'test_avg')] = float(dice_avg.numpy())
            else:
                dice['val_avg'] = float(dice_avg.numpy())
            if saving_eval_dir:
                if os.path.exists(saving_eval_dir):
                    old_dice = load_jason_to_dict(saving_eval_dir)
                    old_dice.update(dice)
                    save_dict_to_json(old_dice, saving_eval_dir)
                else:
                    save_dict_to_json(dice, saving_eval_dir)
        return dice_per_class, dice_avg

    def validate(self):
        start_time = time.time()
        json_save_dir = self.ckpoint_dir + '/' + 'epoch-{}_checkpoint_dice-score.json'.format(self.current_epoch)
        print("mmmm:", json_save_dir)
        dice_per_class, dice_avg = self.eval(self.validation_data_loader, json_save_dir)
        self.scheduler.step(dice_avg)
        is_best = False
        print("best_score:", self.best_score)
        if dice_avg > self.best_score:
            is_best = True
            self.best_score = dice_avg

        print("Validation: Dice Avg: {:.4f} ".format(dice_avg) +
              ' '.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c], dice_per_class[c]) for c in
                        range(self.config["n_classes"] - 1)]) +
              " {:.3f} sec) {}".format(time.time() - start_time,
                                       datetime.datetime.now().strftime("%D %H:%M:%S")))

        # saving model after every setting epochs
        if self.current_epoch % self.config['save_ckpts_epoch_period'] == 0 or self.current_epoch <= 10:
            self.save_checkpoint({'epoch': self.current_epoch,
                                  'model_state_dict': self.model.state_dict(),
                                  'optimizer_state_dict': self.optimizer.state_dict(),
                                  'best_score': self.best_score},
                                 False, self.ckpoint_dir,
                                 prefix='epoch-{}'.format(self.current_epoch))

        self.save_checkpoint({'epoch': self.current_epoch,
                              'model_state_dict': self.model.state_dict(),
                              'optimizer_state_dict': self.optimizer.state_dict(),
                              'best_score': self.best_score},
                             is_best, self.ckpoint_dir, )
    #
    def setup_test_data(self):
    #     testing_data = dataset_1(self.config['testing_list_file'],
    #                              self.config['data_dir'],
    #                              ['with_seg', 'gt'],
    #                              )
    #     self.testing_data_loader = DataLoader(testing_data,
    #                                           batch_size=1,
    #                                           shuffle=False,
    #                                           num_workers=0, )
        self.fmost_testing_data = Dataset(self.config['testing_list_file'])
        self.testing_data_loader = DataLoader(self.fmost_testing_data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0)

    def test(self):
        print('testing...')
        self.setup_model()
        if '.pth' not in self.ckpoint_dir:
            ckpoint_file = os.path.join(self.ckpoint_dir, 'epoch-160_checkpoint.pth')
        else:
            ckpoint_file = self.ckpoint_dir
        print("iiii:", ckpoint_file)
        last_epoch, best_score = self.initialize_model(self.model, optimizer=None, ckpoint_path=ckpoint_file)
        self.setup_test_data()
        # define json save dir for saving validation score
        dir_items = ckpoint_file.split('/')
        json_save_dir = ''
        for dir_item in dir_items[0:-1]:
            json_save_dir += '/' + dir_item
        json_name = dir_items[-1].replace('.pth', '_dice-score_correct_input.json')
        json_save_dir += '/' + json_name
        print("json_save_dir", json_save_dir)
        dice_per_class, dice_avg = self.eval(self.testing_data_loader, json_save_dir)
        print('testing dice:', dice_avg.numpy())
        print('dice_per_class:', dice_per_class.numpy())
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
