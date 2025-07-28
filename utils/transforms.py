import numpy as np
import cv2
import torch
import torch.nn.functional as F

def get_2D_grid(w:int, h:int):
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # hxw
    grid = torch.stack([grid_x, grid_y], dim=0).float()  # grid coordinate, 2xhxw
    return grid

def warp_image(img:torch.Tensor, coord:torch.Tensor):
    '''
    img: (B,) C, H, W  
    coord: (B,) 2, H', W'. h_range: [0, H-1], w_range: [0, W-1]
    '''
    batched = img.ndim == 4
    if not batched:
        img = img.unsqueeze(0)
        coord = coord.unsqueeze(0)
    h_img, w_img = img.shape[-2:]
    scale = torch.tensor([w_img, h_img], dtype=coord.dtype, device=coord.device).view(2,1,1)
    norm_coord = (coord / scale * 2 - 1)
    
    warped = F.grid_sample(img, norm_coord, 'bilinear', padding_mode='zeros', align_corners=False)
    if not batched:
        warped = warped.squeeze(0)
    return warped

def scale_corresp(corresp:torch.Tensor, new_h:int, new_w:int):
    '''corresp: ((B,) 2, H, W)'''
    batched = corresp.ndim == 4
    if not batched:
        corresp = corresp.unsqueeze(0)
    ori_h, ori_w = corresp.shape[-2:]
    scale_h = new_h / ori_h
    scale_w = new_w / ori_w
    scale = torch.tensor([scale_w, scale_h], dtype=corresp.dtype, device=corresp.device).view(2,1,1)
    corresp = F.interpolate(corresp, (new_h, new_w), mode='bilinear', align_corners=False)
    corresp = corresp * scale
    grid = get_2D_grid(new_w, new_h)
    flow = corresp - grid
    if not batched:
        corresp = corresp.squeeze(0)
        flow = flow.squeeze(0)
    return corresp, flow


def apply_colormap(arr:np.ndarray | torch.Tensor, vmin, vmax, cmap=cv2.COLORMAP_JET):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = arr.squeeze()
    assert vmin < vmax
    norm_arr = arr.clip(vmin, vmax)
    norm_arr = np.uint8((arr - vmin) / (vmax - vmin) * 255)
    rgb_img = cv2.applyColorMap(norm_arr, cmap)
    return rgb_img