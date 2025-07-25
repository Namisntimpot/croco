import os
import torch
from torch.utils.data import Dataset
import megfile
import pickle

from dplink import DpflowLinkFactory

try:
    from datasets.ds_config import dataset_root, dataset_names, dataset_nori_index_dir,  batch_size
except:
    from ds_config import dataset_root, dataset_names, dataset_nori_index_dir,  batch_size

class DpLinkDataset(Dataset):
    def __init__(self, link_config:dict, link_name:str='train'):
        super().__init__()
        self.link_config = link_config
        self.link_name = link_name
        self.dataset_names = dataset_names
        
        self.nori_idx_dir = dataset_nori_index_dir
        self.nori_idxes = {}
        for name in self.dataset_names:
            idx_path = os.path.join(self.nori_idx_dir, f"{name}.idx")
            with megfile.smart_open(idx_path, 'rb') as f:
                nori_idx = pickle.load(f)
            self.nori_idxes.update(nori_idx)
        self.len_pairs = len(self.nori_idxes)

        self.link_factory = DpflowLinkFactory(self.link_config)
        self.link = self.link_factory.create_link(self.link_name)

        self.iter = self.link.instance_generator()

    def __len__(self):
        return self.len_pairs
    
    def __getitem__(self, index):
        ret = next(self.iter)
        ret = [torch.from_numpy(r.copy()) for r in ret]
        return tuple(ret)
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    try:
        from datasets.link_config import link_config
    except:
        from link_config import link_config

    import time
    ds = DpLinkDataset(link_config, 'train')
    dataloader = DataLoader(ds, 256, shuffle=True, drop_last=True)
    dataloader = enumerate(dataloader)

    cnt = 0
    t = 0
    num = 0
    while cnt < 20:
        s = time.time()
        idx, ret = next(dataloader)
        e = time.time()
        img1, img2 = ret
        if cnt >= 10:
            t += (e - s)
            num += 1
        print(type(img1), img1.shape, img1.dtype)
        cnt += 1
    print(f"average time: {t / num}, num={num}")