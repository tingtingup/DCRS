import os
import argparse
from segmentation_network import SegmentationExperiment

def config_setting():
    """
    initialize seg training parameters
    """
    log_root = './compare_experiments_1/yznzheng_fmost_ccf_final/segmentation-unet'
    data_root = '/home/amax/disk/tthan/ori_mini_code_from_oyl/data'
    n_classes = 9
    seg_config = dict(
        random_seed=123,
        data='fmost_visor_cat',
        data_dir=os.path.join(data_root, "mindboggle"),
        n_classes=n_classes,
        class_name={k: str(k) for k in range(1, n_classes)},
        learning_rate=1e-3,
        save_ckpts_epoch_period=10,
        begin_epoch=0,
        resume_from_best=False,
        mk_dir=True,
        gen=False,  # used in eval and setup test data
        data_kind=[],
    )
    seg_config['training_list_file'] = os.path.join(data_root, "visor_edt/visor_train_only.list")
    seg_config['validation_list_file'] = os.path.join(data_root, "visor_edt/visor_test_only.list")
    seg_config['testing_list_file'] = os.path.join(data_root, "visor_edt/visor_test_only.list")
    seg_config['log_dir'] = './{}/{}'.format(log_root, seg_config['data'])
    return seg_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-g', default='1', type=str, help='index of used gpu')
    args = parser.parse_args()
    seg_config = config_setting()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    seg_config['n_epochs'] = 300
    seg_config['mk_dir'] = False
    seg_config['test'] = True
    seg_config['test_flag'] = True
    seg_config['train_flag'] = True
    seg_config['data_kind'] = ['gt']
    if not seg_config['mk_dir']:
        seg_config[
            'log_dir'] ='../model/pretrain'
    seg_exp = SegmentationExperiment(seg_config)
    if seg_config['train_flag']:
        seg_exp.train()
    if seg_config['test_flag']:
        seg_exp.test()



if __name__ == '__main__':
    main()
