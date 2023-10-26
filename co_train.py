
import os
import argparse

from Experiments.Regression_Reg_Seg_joint import Regression_Reg_Seg

import shutil



def config_setting():
    """
    initialize regression, reg and seg training parameters
    """
    data_root = './'
    n_classes = 9
    log_root = './'
    regression_config = dict(


        img_path = '/home/amax/disk/tthan/ori_mini_code_from_oyl/data/fmost_visor_cat/concat_fmost_visor.list',
        label_path='/home/amax/disk/tthan/SDF/Fmost_generation_resample/edt',
        label_path_='/home/amax/disk/tthan/SDF/fmost_ants_affine_resample/edt',
        img_path_valadation="/home/amax/disk/tthan/dataset/Fmost/pseudo_by_reg_parameter_affine/val_1_resample_.list",
        label_path_validation='/home/amax/disk/tthan/SDF/fmost_ants_affine_resample/edt',
        label_path_validation_='/home/amax/disk/tthan/SDF/fmost_ants_affine_resample/edt',
        img_path_test="/home/amax/disk/tthan/dataset/Fmost/pseudo_by_reg_parameter_affine/val_1_resample_test_suojian.list",
        label_path_test='/home/amax/disk/tthan/SDF/fmost_ants_affine_resample/edt',
        label_path_test_='/home/amax/disk/tthan/SDF/fmost_ants_affine_resample/edt',

        img_path_1='/home/amax/disk/tthan/ori_mini_code_from_oyl/data/fmost_visor_cat/concat_fmost_visor.list',
        label_path_1='/home/amax/disk/tthan/SDF/visor_new_resample_affine_generation/edt',
        label_path__1='/home/amax/disk/tthan/SDF/visor_new_resample_affine/edt',
        img_path_valadation_1="/home/amax/disk/tthan/ori_mini_code_from_oyl/data/visor_edt/val_1_resample_.list",
        label_path_validation_1='/home/amax/disk/tthan/SDF/visor_new_resample_affine/edt',
        label_path_validation__1='/home/amax/disk/tthan/SDF/visor_new_resample_affine/edt',

        img_path_test_1="/home/amax/disk/tthan/ori_mini_code_from_oyl/data/visor_edt/val_1_resample_test.list",
        label_path_test_1='/home/amax/disk/tthan/SDF/visor_new_resample_affine/edt',
        label_path_test__1='/home/amax/disk/tthan/SDF/visor_new_resample_affine/edt',

        pseudo_data_dir=os.path.join(data_root, 'result_resample'),
        learning_rate=1e-3,
        batch_size=1,
        save_ckpts_epoch_period=10,
        data='train',
        n_epochs=800,
        random_seed=123,
        n_classes=n_classes,
        class_name=[str(k) for k in range(1, n_classes)],
        # -----------------------------need define almost ever time-------------------------------
        begin_epoch=0,  # define model begin epoch, none zero when training from old model
        gen=True,  # used in testing time, define training/testing list when True/False
    )
    regression_config['log_dir'] = './{}/{}'.format(log_root, regression_config['data'])
    reg_config = dict(
        # ------------------------------------no need for change--------------------------------
        random_seed=123,
        data='normalization',
        data_dir=os.path.join(data_root, "ccf-fmost"),
        pseudo_data_dir=os.path.join(data_root, "pseudo_by_reg"),
        n_classes=n_classes,
        class_name=[str(k) for k in range(1, n_classes)],
        # ------------------------------hardly need change-------------------------------------
        # crop_size=[0, 10, 7, 22, 8, 7],  # crop image edges width
        learning_rate=1e-3,
        # learning_rate=0.00025,
        save_ckpts_epoch_period=10,
        # -----------------------------need define almost ever time-------------------------------
        begin_epoch=0,  # define model begin epoch, none zero when training from old model
        # begin_epoch=85,
        resume_from_best=False,  # used in initialize model, define using checkpoint.pth/model_best.pth
        gen_from_best=True,  # used in generate(), define using checkpoint.pth/model_best.pth
        mk_dir=True,
        n_epochs=400,  # 300
        gen=False,  # used in testing time, define training/testing list when True/False
        data_kind=[],  # define which data will used in training
    )
    reg_config['training_list_file'] = os.path.join(data_root, "fmost_edt/train_fmost_ants_affine.txt")
    reg_config['validation_list_file'] = os.path.join(data_root, "fmost_edt/val_fmost_ants_affine.txt")
    reg_config['testing_list_file'] = os.path.join(data_root, "fmost_edt/test_fmost_ants_affine.txt")
    reg_config['log_dir'] = './{}/{}'.format(log_root, reg_config['data'])
    seg_config = dict(
        # -----------------------------------no need for change-----------------------------------
        random_seed=123,
        data='Fmost_segmentation',
        data_dir=os.path.join(data_root, "mindboggle"),
        pseudo_data_dir=os.path.join(data_root, "pseudo_by_seg"),
        n_classes=n_classes,
        class_name={k: str(k) for k in range(1, n_classes)},
        # ---------------------------------hardly need change------------------------------------------
        # crop_size=[0, 10, 7, 22, 8, 7],  # crop image edges width
        learning_rate=1e-3,
        # learning_rate=0.00025,
        save_ckpts_epoch_period=10,
        # -----------------------------------need define almost ever time----------------------------------
        begin_epoch=0,
        # begin_epoch=85,
        resume_from_best=True,
        gen_from_best=True,
        mk_dir=True,
        n_epochs=200,  # 300
        gen=False,  # used in eval and setup test data
        data_kind=[],
        filter=False,
    )
    seg_config['training_list_file'] = os.path.join(data_root, "generated_fmost_edt/train_seg.txt")
    seg_config['validation_list_file'] = os.path.join(data_root, "generated_fmost_edt/val_seg.txt")
    seg_config['testing_list_file'] = os.path.join(data_root, "generated_fmost_edt/test.txt")
    seg_config['log_dir'] = './{}/{}'.format(log_root, seg_config['data'])
    return regression_config, reg_config, seg_config
    # return
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-g', default='0', type=str, help='index of used gpu')
    parser.add_argument('--task', '-t', default=0, type=int, help='index of used gpu')
    args = parser.parse_args()
    regression_config, reg_config, seg_config = config_setting()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.task == 0:
        regression_config['n_epochs'] = 800
        regression_config['mk_dir'] = True
        regression_config['data_kind'] = ['fmost', 'with_seg']
        regression_config['test'] = False
        if not regression_config['mk_dir']:
            regression_config[
                'log_dir'] ='./Architecture'
            regression_config[
                'ckpoint_dir_RS'] = './Architecture'
            regression_config[
                'ckpoint_dir_RSeg'] = './Architecture'
        reg_exp = Regression_Reg_Seg(regression_config)
        reg_exp.train()
        print("ok")
        reg_exp.test()



if __name__ == '__main__':
    main()
    
    
