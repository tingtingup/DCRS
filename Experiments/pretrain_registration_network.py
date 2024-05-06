import os
import argparse
from registration_network import RegistrationExperiment

def config_setting():
    data_root = './'
    n_classes = 9
    reg_config = dict(
        random_seed=123,
        batch_size=1,
        n_classes=n_classes,
        class_name=[str(k) for k in range(1, n_classes)],
        learning_rate=1e-3,
        save_ckpts_epoch_period=10,
        begin_epoch=0,
        resume_from_best=False,
        mk_dir=True,
        n_epochs=160,   ###default (setting)
        gen=False,
        data_kind=[],

    )
    reg_config['training_list_file'] = '/home/amax/disk/tthan/ori_mini_code_from_oyl/data/fmost_visor_cat/concat_fmost_visor.list'
    reg_config['validation_list_file'] = "/home/amax/disk/tthan/dataset/Fmost/pseudo_by_reg_parameter_affine/val_1_resample_.list"
    reg_config['testing_list_file'] = "/home/amax/disk/tthan/dataset/Fmost/pseudo_by_reg_parameter_affine/val_1_resample_test_suojian.list"

    # reg_config['training_list_file'] = os.path.join(data_root, "dataset/train.list")
    # reg_config['validation_list_file'] = os.path.join(data_root, "dataset/val.list")
    # reg_config['testing_list_file'] = os.path.join(data_root, "dataset/test.list")

    return reg_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-g', default='1', type=str, help='index of used gpu')
    args = parser.parse_args()
    reg_config = config_setting()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    reg_config['n_epochs'] = 160
    reg_config['mk_dir'] = False
    reg_config['data_kind'] = ['fmost', 'with_seg']
    reg_config['test_flag'] = True
    reg_config['train_flag'] = True
    if not reg_config['mk_dir']:
        reg_config[
            'log_dir'] ='../model'
    reg_exp = RegistrationExperiment(reg_config)
    if reg_config['train_flag']:
        reg_exp.train()
    if reg_config['test_flag']:
        reg_exp.test()

if __name__ == '__main__':
    main()
