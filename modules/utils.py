import yaml
from functools import reduce
import numpy as np


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def get_ckpt_inf(ckpt_path):
    """get ckpt information"""
    _split_list = ckpt_path.split('e_')[-1].split('_s_')
    epoch = int(_split_list[0])
    step = int(_split_list[-1].split('.ckpt')[0]) + 1

    return epoch, step


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output
