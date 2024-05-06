#!/usr/bin/env python
from base import *
from lib.network_factory import unets
from lib.DATASET_SEG import SegDataSet as dataset
class SegmentationExperiment(BaseExperiment):
    def __init__(self, config):
        super(SegmentationExperiment, self).__init__(config)
        if self.config['mk_dir']:
            day = datetime.datetime.now().strftime('%Y_%m_%d_')
            hour = str(int(datetime.datetime.now().strftime('%H')) + 8)
            minute = datetime.datetime.now().strftime('%M')
            self.date_time = day + hour + ':' + minute
            if len(self.config['data_kind']) == 1:
                if 'gt' in self.config['data_kind']:
                    exp_name = 'Seg_fully-supervised'
                elif 'weak' in self.config['data_kind']:
                    exp_name = 'Seg_weak-data'
            self.exp_name = '{}_epoch-{}'.format(exp_name, self.config['n_epochs'])
            print('exp name:', self.exp_name)
            self.ckpoint_dir = os.path.join(self.config['log_dir'], self.exp_name, self.date_time)
        else:
            self.ckpoint_dir = self.config['log_dir']
            self.config['test'] = self.config['test']
    def setup_log(self):
        if not os.path.isdir(self.ckpoint_dir):
            os.makedirs(self.ckpoint_dir)
        save_dict_to_json(self.config, os.path.join(self.ckpoint_dir, "train_config.json"))
    def setup_model(self):
        # build segmentation model
        self.model = unets.UNet(in_channel=1, n_classes=self.config['n_classes'], bias=True, BN=True).cuda()
        # set up loss
        self.criterion = get_loss_function('dice')(
            n_class=self.config['n_classes'], weight_type='Uniform', no_bg=False, softmax=True, eps=1e-6).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                        patience=10,
                                                        factor=0.5, verbose=True, threshold_mode='abs',
                                                        threshold=0.003,
                                                        min_lr=1e-5)
    def setup_train_data(self):
        print("Initializing dataloader")
        # set up data loader
        training_data = dataset(self.config['training_list_file'],
                                self.config['data_dir'],
                                self.config['data_kind'],
                                )
        validation_data = dataset(self.config['validation_list_file'],
                                  self.config['data_dir'],
                                  ['gt'],
                                  )
        self.training_data_loader = DataLoader(training_data,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=0,
                                               # sampler=training_sampler
                                               )
        self.validation_data_loader = DataLoader(validation_data,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0
                                                 )
    def train(self):
        self.setup_train()
        print("Training {}".format(self.ckpoint_dir))
        if not self.config['resume_from_best']:
            resume_model = os.path.join(self.ckpoint_dir, 'checkpoint.pth')
        else:
            resume_model = os.path.join(self.ckpoint_dir, 'model_best.pth')
        print(resume_model)
        self.best_score = 0
        self.current_epoch = self.config['begin_epoch'] + 1
        print("Start Training:")
        print(self.config['n_epochs'])
        for epoch in range(self.current_epoch, self.config['n_epochs'] + 1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
        print('Finished Training: {}'.format(self.exp_name))

    def train_one_epoch(self):
        running_loss = 0.0
        start_time = time.time()  # log running time
        train_data_iter = iter(self.training_data_loader)
        iters_per_epoch = len(self.training_data_loader.dataset)

        with trange(iters_per_epoch,
                    desc='Epoch[{}/{}] Train'.format(self.current_epoch, self.config['n_epochs'])) as t:
            for i in t:
                self.model.train()
                self.optimizer.zero_grad()
                images, seg, name, truths, sdm = next(train_data_iter)
                output = self.model(sdm.cuda().float())
                # saving onehot
                print ('output:',output.shape)
                print('truth:', truths.shape)
                print(output.type())
                loss = self.criterion(output, truths.cuda().float())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()  # average loss over batches
                duration = time.time() - start_time
                t.set_postfix_str('total_loss: {:.3f}  lr:{} ({:.3f} sec/batch) {}'.format(
                    running_loss,
                    self.optimizer.param_groups[0]['lr'],
                    duration,
                    datetime.datetime.now().strftime("%D %H:%M:%S")
                ))
                running_loss = 0.0
                start_time = time.time()


    def eval(self, dataloader, saving_eval_dir=''):
        with torch.no_grad():
            self.model.eval()
            dice_per_class = torch.tensor([0.] * (self.config["n_classes"] - 1))
            running_dice = torch.zeros(self.config["n_classes"] - 1)
            data_iter = iter(dataloader)
            dice = {}
            dice_dict = {}
            for j in trange(len(dataloader)):
                images, seg, name, truths, sdm = next(data_iter)
                pred = self.model(sdm.cuda().float()).cpu()
                for c in range(1, self.config["n_classes"]):
                    running_dice[c - 1] = metrics.metricEval('dice',
                                                             torch.max(pred.squeeze(), 0)[1].numpy() == c,
                                                             torch.max(truths.squeeze(), 0)[1].numpy() == c,
                                                             num_labels=2)

                    dice_per_class[c - 1] += running_dice[c - 1]
                    dice_dict[c - 1] = float(running_dice[c - 1])

                dice_dict[100] = float(running_dice.mean())
                dice[name[0]] = copy.deepcopy(dice_dict)

            dice_per_class = dice_per_class / (j + 1)
            dice_avg = dice_per_class.mean()
            if len(dataloader.dataset) > 10:
                dice['{}'.format('train_avg' if self.config['gen'] else 'test_avg')] = float(dice_avg.numpy())
            else:
                dice['val_avg'] = float(dice_avg.numpy())
            print(dice)

            if saving_eval_dir:
                print("save_path:", saving_eval_dir)
                if os.path.exists(saving_eval_dir):
                    old_dice = load_jason_to_dict(saving_eval_dir)
                    old_dice.update(dice)
                    save_dict_to_json(old_dice, saving_eval_dir)
                else:
                    print(dice)
                    save_dict_to_json(dice, saving_eval_dir)
        return dice_per_class, dice_avg

    def validate(self):
        # validation
        start_time = time.time()
        ckpoint_file = self.ckpoint_dir
        json_save_dir = ckpoint_file + '/' + 'epoch-{}_checkpoint_dice-score.json'.format(self.current_epoch)
        dice_per_class, dice_avg = self.eval(self.validation_data_loader, json_save_dir)
        self.scheduler.step(dice_avg)
        is_best = False
        if dice_avg > self.best_score:
            is_best = True
            self.best_score = dice_avg

        print("Validation: Dice Avg: {:.4f} ".format(dice_avg) +
              ' '.join(["Dice_{}:{:.3f}".format(self.config["class_name"][c + 1], dice_per_class[c]) for c in
                        range(self.config["n_classes"] - 1)]) +
              " {:.3f} sec) {}".format(time.time() - start_time,
                                       datetime.datetime.now().strftime("%D %H:%M:%S")))

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

    def setup_test_data(self):
        testing_data = dataset(self.config['testing_list_file'] if not self.config['gen']
                               else self.config['training_list_file'],
                               self.config['data_dir'],
                               ['gt'],
                               )

        self.testing_data_loader = DataLoader(testing_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0
                                              )

    def test(self):
        print('testing...')
        self.setup_model()
        if '.pth' not in self.ckpoint_dir:
            ckpoint_file = os.path.join(self.ckpoint_dir, 'segmentation.pth')
        else:
            ckpoint_file = self.ckpoint_dir
        last_epoch, best_score = self.initialize_model(self.model, optimizer=None, ckpoint_path=ckpoint_file)
        self.setup_test_data()
        dir_items = ckpoint_file.split('/')
        json_save_dir = ''
        for dir_item in dir_items[1:-1]:
            json_save_dir += '/' + dir_item
        json_name = dir_items[-1].replace('.pth', '_dice-score.json')
        json_save_dir += '/' + json_name
        dice_per_class, dice_avg = self.eval(self.testing_data_loader, json_save_dir)
        print(dice_avg.numpy())
        return float(dice_avg.numpy())
