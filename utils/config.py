import os
from omegaconf import OmegaConf
from easydict import EasyDict

def __load_yaml_leaf(path):
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg)
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

def __override_model_kwargs(cfg:dict):
    if "override_kwargs" in cfg['model']:
        for k, v in cfg['model']["override_kwargs"].items():
            cfg['model']['kwargs'][k] = v
    return cfg


def load_config(path):
    '''recursively load all .yaml files.'''
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg)
    cfg = __recursive_load_yaml(cfg)
    cfg = __override_model_kwargs(cfg)
    return cfg


def args_parse_bool_str(args):
    for k, v in vars(args).items():
        if not isinstance(v, str):
            continue
        if v.lower() == 'true':
            setattr(args, k, True)
        elif v.lower() == 'false':
            setattr(args, k, False)
    return args