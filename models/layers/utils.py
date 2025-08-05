import torch
import torch.nn as nn
import collections.abc
from itertools import repeat

def parse_norm_layer_1d(norm_layer:str, dim):
    if norm_layer is None:
        return nn.Identity()
    norm_layer = norm_layer.lower()
    if norm_layer == 'layer_norm':
        return nn.LayerNorm(dim, eps=1e-6)
    elif norm_layer == 'batch_norm':
        return nn.BatchNorm1d(dim, eps=1e-6)
    elif norm_layer == 'instance_norm':
        return nn.InstanceNorm1d(dim, eps=1e-6)
    else:
        return nn.Identity()
    

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)  
    p: int or (int, int), patch_size, (h, w)
    x: (N, L, patch_size**2 *3)
    """
    p = to_2tuple(p)
    assert imgs.shape[2] % p[0] == 0 and imgs.shape[3] % p[1] == 0

    h = imgs.shape[2] // p[0]
    w = imgs.shape[3] // p[1]
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p[0], w, p[1]))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p[0] * p[1] * 3))
    return x

def unpatchify(x, p, ori_reso):
    """
    x: (N, L, patch_size**2 *3)  
    p: int or (int, int), patch_size, (h, w)  
    ori_reso: int or (int, int), original image resolution, (h, w)
    imgs: (N, 3, H, W)  
    """
    p = to_2tuple(p)
    ori_reso = to_2tuple(ori_reso)
    assert x.shape[2] == p[0] * p[1] * 3
    num_patches_h = ori_reso[0] // p[0]
    num_patches_w = ori_reso[1] // p[1]
    assert num_patches_h * num_patches_w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], num_patches_h, num_patches_w, p[0], p[1], 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, num_patches_h * p[0], num_patches_w * p[1]))
    return imgs