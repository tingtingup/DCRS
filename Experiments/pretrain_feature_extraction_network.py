import os
from feature_extration_network import G


def config_setting():
    """
    initialize training parameters
    """
    log_root = 'MODEL_TTHAN'
    G_config = dict(
        train_dataset='/home/amax/disk/tthan/ori_mini_code_from_oyl/data/visor_edt/brain.generation_random250.list',
        validation_dataset="/home/amax/disk/tthan/ori_mini_code_from_oyl/data/visor_edt/brain_val.list",
        test_dataset="/home/amax/disk/tthan/DCRS-main/dataset/brain_test.list",
        learning_rate=0.001,
        batch_size=1,
        save_ckpts_epoch_period=10,
        data='mindboggle',
        n_epochs=500,
        random_seed=123,
        # -----------------------------need define almost ever time-------------------------------
        begin_epoch=0,  # define model begin epoch, none zero when training from old model
        resume_from_best=False,  # used in initialize model, define using checkpoint.pth/model_best.pth
    )
    G_config['log_dir'] = './{}/{}'.format(log_root, G_config['data'])
    return G_config

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    G_config = config_setting()
    G_config['mk_dir'] = True
    G_config['train_flag'] = True
    G_config['test_flag'] = True
    # G_config['test'] = False
    if not G_config['mk_dir']:
        G_config['log_dir'] = '../model/pretrain'
    reg_exp = G(G_config)
    if G_config['train_flag']:
        reg_exp.train()
    if G_config['test_flag']:
        reg_exp.test()
if __name__ == '__main__':
    main()



