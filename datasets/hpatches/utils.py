import numpy as np
import torch


def convert_mapping_to_flow(mapping:torch.Tensor):
    '''
    mapping: (B,)*2*H*W  
    output_flow: (B,)*2*H*W
    '''
    batched = (mapping.ndim == 4)
    if not batched:
        mapping = mapping.unsqueeze(0)
    B,C,H,W = mapping.shape
    xx = torch.arange(0, W, device=mapping.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=mapping.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    flow = mapping.float() - grid
    if not batched:
        flow = flow.squeeze(0)
    return flow