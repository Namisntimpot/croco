import os
from PIL import Image
from dplink import Link
import megfile 
import pickle
import random
import numpy as np
import nori2
from io import BytesIO

try:
    from ds_config import dataset_root, dataset_names, dataset_nori_index_dir, dataset_nori_volume_dir, transforms_operations, batch_size, seed
    from transforms import get_pair_transforms
except:
    from datasets.ds_config import dataset_root, dataset_names, dataset_nori_index_dir, dataset_nori_volume_dir, transforms_operations, batch_size, seed
    from datasets.transforms import get_pair_transforms


class NidLink(Link):
    def __init__(self):
        super().__init__()
        self.ds_root = dataset_root
        self.ds_names = dataset_names
        self.nori_idx_dir = dataset_nori_index_dir
        self.nori_idxes = {}

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        for name in self.ds_names:
            idx_path = os.path.join(self.nori_idx_dir, f"{name}.idx")
            with megfile.smart_open(idx_path, 'rb') as f:
                nori_idx = pickle.load(f)
            print(f"{name}: {len(nori_idx)} pairs")
            self.nori_idxes.update(nori_idx)

        self.len_pairs = len(self.nori_idxes)
        print(f"total: {self.len_pairs} pairs.")

    def instance_generator(self):
        _cnt = 0
        while True:
            if _cnt == 0:  # start.
                keys = list(self.nori_idxes.keys())
                random.shuffle(keys)
            cur_key = keys[_cnt]
            _cnt += 1
            if _cnt >= self.len_pairs:
                _cnt = 0
            cur_pair = self.nori_idxes[cur_key]
            nid_img_1 = cur_pair['img_1']
            nid_img_2 = cur_pair['img_2']
            yield nid_img_1, nid_img_2


class ProcessLink(Link):
    def __init__(self):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.transforms = get_pair_transforms(transforms_operations, totensor=True, normalize=True)
        self.batch_size = batch_size

    def _load_rgb_img(self, b1, b2):
        stream1 = BytesIO(b1)
        stream2 = BytesIO(b2)
        img1 = Image.open(stream1).convert("RGB")
        img2 = Image.open(stream2).convert("RGB")
        img1, img2 = self.transforms(img1, img2)
        stream1.close()
        stream2.close()
        return img1.numpy(), img2.numpy()

    def instance_generator(self):
        nid_generator = self.remote_instance_generator('nid')
        fetcher = nori2.Fetcher()
        while True:
            # _cnt = 0
            # img1s = []
            # img2s = []
            for nid_1, nid_2 in nid_generator:
                # _cnt += 1
                img1 = fetcher.get(nid_1)
                img2 = fetcher.get(nid_2)
                img1, img2 = self._load_rgb_img(img1, img2)
                # img1s.append(img1)
                # img2s.append(img2)
                yield img1, img2


if __name__ == "__main__":
    import argparse
    from dplink import LocalLinkFactory, DpflowLinkFactory
    from link_config import link_config
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action='store_true')
    args = parser.parse_args()

    if not args.remote:
        link_factory = LocalLinkFactory(link_config)
        link = link_factory.create_link("nid")
        for nid1, nid2 in link.instance_generator():
            print(nid1, nid2)
            break
    else:
        link_factory = DpflowLinkFactory(link_config)
        link = link_factory.create_link("train")
        for img1, img2 in link.instance_generator():
            print(type(img1), type(img2), img1.shape, img2.shape, img1.dtype)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
            for img, name in zip([img1, img2], ["img1.jpg", "img2.jpg"]):
                img = img * std + mean
                img = (np.transpose(img, (1,2,0)) * 255).astype(np.uint8)
                cv2.imwrite(name, img)
                break