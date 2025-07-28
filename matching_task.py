import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import reduce
from collections import OrderedDict
import megfile
import json
import cv2
import imageio

from models import make_model
from models.criterion import flow_matrics
from datasets.hpatches.hpatches_dataset import HPatchesDataset, DirectoryDataset
from utils.config import load_config
from utils.transforms import scale_corresp, apply_colormap, warp_image
from utils.misc import make_directories

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_path", type=str, help="path to the config yaml file.")
    parser.add_argument("-p", "--parameter", type=str, help="path to the pretrained parameter file to overwrite what the config specifies."
                        "leave it None to use the parameter file specified in the config.", default=None)
    parser.add_argument("--hpatches_root", type=str, default=None, help="Root directory of HPatches benchmark. If specified, the script will run hpatches benchmark.")
    parser.add_argument("-src", "--source_path", type=str, default=None, help="path to the source image/folder." 
                        "If hpatches_root is None, the script will estimate flow between the custom source image(s) and target image(s) specified by source_path and target_path.")
    parser.add_argument("-trg", "--target_path", type=str, default=None, help="path to the target image/folder."
                        "If hpatches_root is None, the script will estimate flow between the custom source image(s) and target image(s) specified by source_path and target_path.")
    parser.add_argument("-flow", "--flow_path", type=str, default=None, help="There might be (pseudo) gt flow (e.g., through structured light decoding)."
                        "they must be .npy files containing H*W*2 array. Large values indicate invalid pixels.")
    
    parser.add_argument("--output_path", type=str, help="path to the output directory.")
    parser.add_argument("-o_attn", "--save_attn_map", action='store_true', help="whether to save attention map.")
    parser.add_argument("-o_flow", "--save_flow", action="store_true", help="whether to save flow data.")
    parser.add_argument("-o_warp", "--save_warp", action='store_true', help="whether to save warped image.")

    parser.add_argument("--reciprocity", type=bool, default=None, help="reciprocity. If not None, it is used to overwrite what the config specifies.")
    parser.add_argument("--suppress_1st_token", type=bool, default=None, help="whether to suppress the attention weight of the 1st token."
                        "If not None, it will overwrite what the config file specifies.")
    parser.add_argument("--attn_layers_adopted", type=str, default=None, help="Which attention layers should be used for construct correlation volume."
                        "It should be comma seperated string of numbers. If not None, it will overwrite what the config file specifies.")
    parser.add_argument("--temperature", type=float, default=None, help="temperature. If not None, it will overwrite the config.")
    parser.add_argument("--img_h", type=int, default=None, help="input image height. If not None, it will overwrite the config.")
    parser.add_argument("--img_w", type=int, default=None, help="input image width.  If not None, it will overwrite the config.")
    args = parser.parse_args()

    config = load_config(args.config_path)
    # overwrite
    config.model.pretrained_ckpt = args.parameter if args.parameter is not None else config.model.pretrained_ckpt
    config.inference.reciprocity = args.reciprocity if args.reciprocity is not None else config.inference.reciprocity
    config.inference.suppress_1st_token = args.suppress_1st_token if args.suppress_1st_token is not None else config.inference.suppress_1st_token
    config.inference.attn_layers_adopted = args.attn_layers_adopted if args.attn_layers_adopted is not None else config.inference.attn_layers_adopted
    config.inference.temperature = args.temperature if args.temperature is not None else config.inference.temperature
    config.data.img_size = (args.img_h, args.img_w) if args.img_h is not None and args.img_w is not None else config.data.img_size

    args.config = config
    return args

def save_attention_map(attn_dir:str, key:str, src_img:torch.Tensor, trg_img:torch.Tensor,
                       attn_map:torch.Tensor, source_path:torch.Tensor, target_path:torch.Tensor, suffix:str=""):
    '''
    src_img, trg_img: torch.Tensor[C*H*W]  
    attn_map: torch.Tensor[num_patches * num_patches]  
    '''
    hsrc, wsrc = src_img.shape[-2:]
    htrg, wtrg = trg_img.shape[-2:]
    n_patch_src, n_patch_trg = attn_map.shape
    patch_size = int(np.sqrt(hsrc*wsrc / n_patch_src))
    h_patch_src, w_patch_src = hsrc / patch_size, wsrc / patch_size
    h_patch_trg, w_patch_trg = htrg / patch_size, wtrg / patch_size
    attn_map = attn_map.reshape(h_patch_src, w_patch_src, h_patch_trg, w_patch_trg).detach().cpu().numpy()
    attn_data = {
        "source_path": source_path, "target_path": target_path, "attn_map": attn_map
    }
    suffix = "_" + suffix if len(suffix) > 0 and not suffix.startswith("_") else suffix
    path = os.path.join(attn_dir, f"{key}{suffix}.npy")
    with megfile.smart_open(path, 'wb') as f:
        np.save(f, attn_data)


def save_flow(flow_dir: str, key:str, flow:torch.Tensor, err_dir:str=None, flow_gt:torch.Tensor=None, suffix=""):
    '''flow: 2*H*W'''
    vmin, vmax = -20, 20
    suffix = "_" + suffix if len(suffix) > 0 and not suffix.startswith("_") else suffix
    flow_path = os.path.join(flow_dir, f"{key}{suffix}.npy")
    err_path = None if err_dir is None or flow_gt is None else os.path.join(err_dir, f"{key}{suffix}.jpg")
    flow_np = flow.squeeze().detach().cpu().numpy()
    with megfile.smart_open(flow_path, 'wb') as f:
        np.save(f, flow_np)
    if flow_gt is not None and err_path is not None:
        err = torch.sum((flow.squeeze() - flow_gt.squeeze()) ** 2, dim=0).sqrt().detach().cpu().numpy()
        err_img = apply_colormap(err, vmin, vmax)
        with megfile.smart_open(err_path, "wb") as f:
            imageio.imwrite(f, err_img, format='jpg')


def save_warped_img(warp_dir:str, key:str, corresp:torch.Tensor, src_img:torch.Tensor, trg_img:torch.Tensor, suffix="", normalized=True):
    '''src_img, trg_img: C*H*W, corresp: 2*H*W
       if normalized, the image should be unnormalized before being saved.'''
    suffix = "_" + suffix if len(suffix) > 0 and not suffix.startswith("_") else suffix
    warp_path = os.path.join(warp_dir, f"{key}{suffix}.jpg")
    
    warped_image = warp_image(trg_img, corresp)
    if normalized:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=warped_image.device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=warped_image.device).view(3,1,1)
        warped_image = warped_image * std + mean
        src_img = src_img * std + mean
    src_img = np.uint8(src_img.detach().squeeze().permute(1,2,0).cpu().numpy())
    warped_image = np.uint8(warped_image.detach().squeeze().permute(1,2,0).cpu().numpy() * 255)
    cat = np.concatenate([src_img, warped_image], axis=1)
    with megfile.smart_open(warp_path, 'wb') as f:
        imageio.imwrite(warp_path, cat, 'jpg')


def matching_task(model:torch.nn.Module, dataset:torch.utils.data.Dataset, args):
    data_loader = DataLoader(dataset, 1, False)
    iterator = tqdm(enumerate(data_loader), total=len(data_loader))

    has_gt = dataset.has_gt

    attn_dir = os.path.join(args.output_path, 'attn_maps') if args.save_attn_map else None
    flow_dir = os.path.join(args.output_path, 'flow') if args.save_flow else None
    err_dir = os.path.join(args.output_path, 'error') if args.save_flow and has_gt else None
    warp_dir = os.path.join(args.output_path, 'warp') if args.save_warp else None
    make_directories(attn_dir, flow_dir, err_dir, warp_dir)
    save_info = reduce(lambda x,y: x or y, [d is not None for d in [attn_dir,flow_dir,err_dir,warp_dir]])


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    _cnt = 0
    total_matrics = {}
    info = OrderedDict()

    if not has_gt and not args.save_attn_map and not args.save_flow and not args.save_warp:
        raise ValueError("The specified source_path and target_path don't have ground-truth flow,"
                         "and nothing is specified to be saved. The script won't produce anything and stops here.")
    
    for idx, data in iterator:
        key:str = data['key'][0]
        src_img:torch.Tensor = data['source_image'].to(device)
        trg_img:torch.Tensor = data['target_image'].to(device)
        src_path:str = data['source_path'][0]
        trg_path:str = data['target_path'][0]
        h, w = src_img.shape[-2:]

        info[key] = {"source_path": os.path.realpath(src_path), "target_path": os.path.realpath(trg_path)}
        
        has_gt = 'flow' in data
        if has_gt:
            flow:torch.Tensor = data['flow'].to(device)
            corresp: torch.Tensor = data['corresp'].to(device)
            mask:torch.Tensor = data['mask'].to(device)

        out = model.forward(src_img, trg_img, **args.inference)
        attn_maps, corresp_pred, flow_pred = out[-3:]
        h_pred, w_pred = corresp_pred.shape[-2:]
        if h_pred != h or w_pred != w:
            corresp_pred, flow_pred = scale_corresp(corresp_pred, h, w)
        if has_gt:
            mat = flow_matrics(flow_pred, flow, mask)
            mat_str = f"aepe_{mat['aepe']:.3f}"
            _cnt += 1
            mat = {k: v.item() for k, v in mat.items()}
            for k, v in mat.items():
                total_matrics[k] = total_matrics.get(k, 0) + v
            info[key]['matrics'] = mat
        else:
            mat_str = ""

        attn_maps = attn_maps.squeeze_(0)  # num_patches(img1) * num_patches(img2)
        corresp_pred = corresp_pred.squeeze_(0)
        flow_pred = flow_pred.squeeze_(0)
        
        if attn_dir is not None:
            save_attention_map(attn_dir, key, src_img, trg_img, attn_maps, src_path, trg_path, mat_str)
        if  flow_dir is not None:
            save_flow(flow_dir, key, flow_pred, err_dir, flow if has_gt else None, mat_str)
        if warp_dir is not None:
            save_warped_img(warp_dir, key, corresp_pred, src_img.squeeze(), trg_img.squeeze(), mat_str)
        
    if save_info:
        with megfile.smart_open(os.path.join(args.output_path, "info.json"), 'w') as f:
            json.dump(info, f, indent=4)
    total_matrics = {k: v / _cnt for k, v in total_matrics.items()}
    return total_matrics


def main():
    args = parse_args()
    config = args.config

    model_config = config.model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # initialize model
    model:torch.nn.Module = make_model(model_config.task, model_config.pretrain_type, **model_config.kwargs)
    ckpt = torch.load(model_config.pretrained_ckpt, map_location='cpu')
    ckpt = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # create directorys
    make_directories(args.output_path)

    # initialize dataset and dataloader
    data_config = config.data
    if args.hpatches_root is not None:
        # hpatches benchmark
        ret = {}
        original_output_path = args.output_path
        for pair_name, path_list_csv in data_config.hpatches_list_csv.items():
            args.output_path = os.path.join(original_output_path, pair_name)
            make_directories(args.output_path)
            print(f"process pair: {pair_name}")
            ds = HPatchesDataset(args.hpatches_root, path_list_csv, data_config.img_size, True)
            pair_ret = matching_task(model, ds, args)
            ret[pair_name] = pair_ret
            for k, v in ret.items():
                print(f"{k}: {v}")

        with megfile.smart_open(os.path.join(original_output_path, 'matrics.json'), 'w') as f:
            json.dump(ret, f, indent=4)

    else:
        # custom source and target images.
        ds = DirectoryDataset(args.source_path, args.target_path, data_config.img_size, True)
        ret = matching_task(model, ds, args)
        if len(ret) != 0:
            for k, v in ret.items():
                print(f"{k}: {v}")
        

if __name__ == '__main__':
    with torch.no_grad():
        main()