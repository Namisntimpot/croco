import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from glob import glob
from torch.nn.functional import interpolate

try:
    from .utils import convert_mapping_to_flow
except:
    from datasets.hpatches.utils import convert_mapping_to_flow

def get_grid(w:int, h:int):
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # hxw
    grid = torch.stack([grid_x, grid_y], dim=0).float()  # grid coordinate, 2xhxw
    return grid

class HPatchesDataset(Dataset):
    def __init__(self, root, path_list_csv, image_size=(240, 240), normalize=True):
        '''
        Args:  
        root: hpatches dataset root dir
        path_list_csv: path to csv file with ground-truth data information  
        image_size: image size (H,W) used for evaluation. If None, it will use original resolution. Default=(240, 240) as in DGC-Net  
        normalize: whether applying normalization ((img-mean) / std). if False, only scale images to range (0,1)  
        '''
        super().__init__()
        self.root = root
        self.path_list_csv = path_list_csv
        self.image_size = image_size
        self.df = pd.read_csv(path_list_csv)
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)
        self.normalize = normalize

        self.has_gt = True

    def __len__(self):
        return len(self.df)

    def _normalize_img(self, img:torch.Tensor):
        '''img: CHW, range (0, 1)'''
        return (img - self.mean) / self.std
    
    def __getitem__(self, index):
        """
        Returns: Dictionary with fieldnames:
                source_image: C*H*W
                target_image: C*H*W
                flow: 2*H*W
                corresp: 2*H*W, correspondence map.  
                homography: 3*3, homography matrix.
                mask: H*W, valid map of the flow/corresp
        """
        data = self.df.iloc[index]
        obj = str(data.obj)  # directory name.
        obj_dir = os.path.join(self.root, obj)
        im1_id, im2_id = str(data.im1), str(data.im2)
        src_path = os.path.join(obj_dir, im1_id+".ppm")
        trg_path = os.path.join(obj_dir, im2_id+".ppm")
        key = f"{obj}_{im1_id}_{im2_id}"

        h_ref_orig, w_ref_orig = data.Him.astype("int"), data.Wim.astype("int")

        # load image
        src_img = cv2.imread(src_path)
        trg_img = cv2.imread(trg_path)
        if src_img.shape[-1] == 3:
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            trg_img = cv2.cvtColor(trg_img, cv2.COLOR_BGR2RGB)

        h_trg_orig, w_trg_orig = trg_img.shape[:2]
        if self.image_size is None:
            h_scale, w_scale = h_trg_orig, w_trg_orig
        else:
            h_scale, w_scale = self.image_size
            src_img = cv2.resize(src_img, (w_scale, h_scale))
            trg_img = cv2.resize(trg_img, (w_scale, h_scale))
        src_img = torch.from_numpy(src_img).float().permute(2,0,1) / 255.  # C,H,W
        trg_img = torch.from_numpy(trg_img).float().permute(2,0,1) / 255.
        if self.normalize:
            src_img = self._normalize_img(src_img)
            trg_img = self._normalize_img(trg_img)

        # homography matrix
        H = data[5:].astype('double').values.reshape((3,3))
        # As gt homography is calculated for (h_orig, w_orig) images,
        # we need to
        # map it to (h_scale, w_scale), that is 240x240
        # H_scale = S * H * inv(S)
        S1 = np.array([[w_scale / w_ref_orig, 0, 0],
                       [0, h_scale / h_ref_orig, 0],
                       [0, 0, 1]])
        S2 = np.array([[w_scale / w_trg_orig, 0, 0],
                       [0, h_scale / h_trg_orig, 0],
                       [0, 0, 1]])
        H_scale = np.dot(np.dot(S2, H), np.linalg.inv(S1))
        # inverse homography matrix
        Hinv = np.linalg.inv(H_scale)

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        Xwarp=XwarpHom / (ZwarpHom + 1e-8)
        Ywarp=YwarpHom / (ZwarpHom + 1e-8)
        # and now the grid
        grid_gt = torch.stack([Xwarp.view(h_scale, w_scale),
                               Ywarp.view(h_scale, w_scale)], dim=0)  # 2,h,w
        flow_gt = convert_mapping_to_flow(grid_gt)  # 2,h,w

        # mask
        mask = grid_gt[0].ge(0) & grid_gt[0].le(w_scale-1) & \
               grid_gt[1].ge(0) & grid_gt[1].le(h_scale-1)
        
        ret = {
            'key': key,
            'source_image': src_img,
            'target_image': trg_img,
            'flow': flow_gt,
            'corresp': grid_gt,
            'mask': mask,
            'homography': H_scale,
            'source_path': src_path,
            'target_path': trg_path,
        }
        return ret
    

class DirectoryDataset(Dataset):
    def __init__(self, source_path:str, target_path: str, flow_path:str=None, image_size=(240, 240), normalize=True):
        '''
        image_size: image size (H,W) used for evaluation. If None, it will use original resolution. Default=(240, 240) as in DGC-Net  
        normalize: whether applying normalization ((img-mean) / std). if False, we only scale images to range (0,1)  
        '''
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.flow_path = flow_path
        self.has_gt = flow_path is not None
        assert os.path.exists(self.source_path) and os.path.exists(self.target_path)
        if os.path.isfile(self.source_path):
            assert os.path.isfile(self.target_path), f"{target_path} should be a file as {source_path}."
            assert flow_path is None or os.path.isfile(flow_path), f"flow_path must either be None or a file."
            self.source_path = [self.source_path]
            self.target_path = [self.target_path]
            self.flow_path = [self.flow_path]
        else:
            assert os.path.isdir(self.target_path), f"{target_path} should be a directory as {source_path}."
            assert flow_path is None or os.path.isdir(flow_path), f"flow_path must either be None or a directory."
            self.source_path = sorted(glob(os.path.join(self.source_path, '*')))
            self.target_path = sorted(glob(os.path.join(self.target_path, '*')))
            self.flow_path = [ None for _ in range(len(self.source_path))] if self.flow_path is None else \
                             sorted(glob(os.path.join(self.flow_path, '*')))

        self.image_size = image_size
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)
        self.normalize = normalize

    def _normalize_img(self, img:torch.Tensor):
        '''img: CHW, range (0, 1)'''
        return (img - self.mean) / self.std
    
    def __len__(self):
        return len(self.source_path)
    
    def __getitem__(self, index):
        src_path = self.source_path[index]
        trg_path = self.target_path[index]
        flow_path = self.flow_path[index]
        key = os.path.splitext(os.path.basename(src_path))[0]

        src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
        trg_img = cv2.cvtColor(cv2.imread(trg_path), cv2.COLOR_BGR2RGB)
        if self.image_size is None:
            h, w = src_img.shape[:2]
        else:
            h, w = self.image_size
            src_img = cv2.resize(src_img, (w, h))
            trg_img = cv2.resize(trg_img, (w, h))
        src_img = torch.from_numpy(src_img).float().permute(2,0,1) / 255.
        trg_img = torch.from_numpy(trg_img).float().permute(2,0,1) / 255.
        if self.normalize:
            src_img = self._normalize_img(src_img)
            trg_img = self._normalize_img(trg_img)

        ret = {
            "key": key,
            "source_image": src_img,
            "target_image": trg_img,
            "source_path": src_path,
            "target_path": trg_path,
        }

        if flow_path is not None:
            flow = np.load(flow_path)  # H*W*2
            flow = torch.from_numpy(flow).float().permute(2,0,1)
            if h != flow.shape[1] or w != flow.shape[2]:
                h_scale = 1. * h / flow.shape[1]
                w_scale = 1. * w / flow.shape[2]
                scale = torch.tensor([w_scale, h_scale], dtype=flow.dtype).view(2,1,1)
                flow = interpolate(flow.unsqueeze_(0), (h, w), mode='bilinear', align_corners=False).squeeze_(0)
                flow = flow * scale
            ret['flow'] = flow
            grid = get_grid(w, h)
            corresp = grid + flow
            ret['corresp'] = corresp
            ret['mask'] = (corresp[0].ge(0) & corresp[0].le(w - 1) & corresp[1].ge(0) & corresp[1].le(h - 1))

        return ret