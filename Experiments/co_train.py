import os
import argparse
from Experiments.Regression_Reg_Seg_joint import Regression_Reg_Seg
import shutil

def config_setting():
    """
    initialize training parameters
    """
    ##### initialize feature extraction network(G), registration(R) and segmentation(S) training parameters
    data_root = './'
    n_classes = 9
    log_root = './'
    G_config = dict(
        data = 'log',
        test_save_data_dir=os.path.join(data_root, 'test_result'),
        learning_rate=1e-3,
        batch_size=1,
        save_ckpts_epoch_period=10,
        save_dir_name ='train',
        n_epochs=800,
        random_seed=123,
        n_classes=n_classes,
        class_name=[str(k) for k in range(1, n_classes)],
        begin_epoch=0,  # define model begin epoch, none zero when training from old model
        gen=True,  # used in testing time, define training/testing list when True/False
    )
    G_config['training_list_file'] = '/home/amax/disk/tthan/ori_mini_code_from_oyl/data/fmost_visor_cat/concat_fmost_visor.list'
    G_config['validation_list_file'] = "/home/amax/disk/tthan/dataset/Fmost/pseudo_by_reg_parameter_affine/val_1_resample_.list"
    G_config['testing_list_file'] = "/home/amax/disk/tthan/dataset/Fmost/pseudo_by_reg_parameter_affine/val_1_resample_test_suojian.list"
    G_config['log_dir'] = './{}/{}'.format(log_root, G_config['data'])
    return G_config
    # return
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-g', default='0', type=str, help='index of used gpu')
    args = parser.parse_args()
    G_config = config_setting()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    G_config['n_epochs'] = 800
    G_config['mk_dir'] = True
    G_config['data_kind'] = ['fmost', 'with_seg']
    G_config['test_flag'] = True
    G_config['train_flag'] = True
    if not G_config['mk_dir']:
        G_config[
            'log_dir'] ='../model'
        G_config[
            'ckpoint_dir_RS'] = '../model'
        G_config[
            'ckpoint_dir_RSeg'] = '../model'
    reg_exp = Regression_Reg_Seg(G_config)
    if G_config['train_flag']:
        reg_exp.train()
        print("ok")
    if G_config['test_flag']:
        reg_exp.test()

if __name__ == '__main__':
    main()
    
    
