#%%
from utils.GLOBAL import ROOT_PATH
from utils import load_vllm_editor
import os, argparse

import yaml

import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


def get_attr():
    def parse_lkpt(value:str):
        if value.lower() == 'none':
            return None 
        return value
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: llava...', required=True)
    parser.add_argument('-dna', '--data_name', type=str, help = 'Train dataset, including EVQA, EIC.', required = True)
    parser.add_argument('-bs', '--batch_size', type=int, help = 'Train dataset sample number.', required = True)
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    # other settings
    parser.add_argument('-dn', '--data_n', type=int, default=None, help = 'Train dataset sample number.')
    parser.add_argument('-lkpt', '--load_ckpt_path', type=parse_lkpt, default = None, help='Editor checkpoint path.')
    parser.add_argument('-edvc', '--extra_devices', type=int, nargs='+', default = [0], help='Extra CUDA devices, default empty.')
    parser.add_argument('-eps', '--epochs', type=int, default=1000, help = 'Train epochs.')
    parser.add_argument('-tnp', '--train_name_prefix', type=str, default=None, help = 'Train name prefix.')
    parser.add_argument('-sci', '--save_ckpt_per_i', type=int, default=1000, help = 'Save checkpoint per iteraions.')
    parser.add_argument('-lpi', '--log_per_i', type=int, default=1, help = 'Log per iteraions.')
    parser.add_argument('-ea', '--ema_alpha', type=float, default=0.1, help = 'EMA loss alpha.')
    parser.add_argument('-rs', '--random_seed', type=int, default=None, help = 'Random seed.')
    parser.add_argument('-dbs', '--data_buffer_size', type=int, default=4, help = 'Buffer size of data generator.')

    parser.add_argument('-el', '--edit_layer', type=int, default=18, help = 'Edit Layer.')

    parser.add_argument('-cp', '--config_path', type=str, default=None, help = 'Config path.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg = get_attr()
    cfg.data_name = cfg.data_name.upper()

    if cfg.config_path is not None:
        # read yaml file
        with open(os.path.join(ROOT_PATH, cfg.config_path), 'r') as f:
            train_cfg = yaml.safe_load(f)
    else:
        train_cfg = None

    # load editor
    editor = load_vllm_editor('vead', cfg.edit_model_name, cfg.device, cfg.extra_devices, None, True, cfg.config_path)
    # load data
    if cfg.data_name == 'EVQA':
        from dataset.vllm import EVQA
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/vqa/vqa_train.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm/images')
        train_data = EVQA(data_path, img_root_dir, cfg.data_n)
    elif cfg.data_name == 'EIC':
        from dataset.vllm import EIC
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/caption/caption_train_edit.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm/images')
        train_data = EIC(data_path, img_root_dir, cfg.data_n)
    # initialize and train
    editor.train_init(train_data, cfg.batch_size, train_name_prefix = cfg.train_name_prefix,
        load_ckpt_path = cfg.load_ckpt_path, save_ckpt_per_i = cfg.save_ckpt_per_i, 
        log_per_i = cfg.log_per_i, ema_alpha = cfg.ema_alpha, random_seed = cfg.random_seed,
        data_buffer_size = cfg.data_buffer_size, edit_model_name = cfg.edit_model_name, train_cfg = train_cfg, dataset_name = cfg.data_name)
    editor.train(cfg.epochs)
# %%
