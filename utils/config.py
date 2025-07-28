import os
from omegaconf import OmegaConf
from easydict import EasyDict

def __load_yaml_leaf(path):
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg

def __recursive_load_yaml(cfg:dict):
    update = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v.endswith(".yaml") and os.path.isfile(v):
            v = __load_yaml_leaf(v)
            v = __recursive_load_yaml(v)
        elif hasattr(v, 'items'):
            v = __recursive_load_yaml(v)
        update[k] = v
    return EasyDict(update)

def load_config(path):
    '''recursively load all .yaml files.'''
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return __recursive_load_yaml(cfg)