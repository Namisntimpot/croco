import numpy
from PIL import Image
from dplink import Link, DpflowLink
import megfile 
import pickle

from ds_config import dataset_root, dataset_names, dataset_nori_index_dir, dataset_nori_volume_dir

dataset_names = ['habitat_release']

class NidLink(Link):
    def __init__(self):
        super().__init__()
        self.ds_root = dataset_root
        self.ds_names = dataset_names
        self.nori_idx_dir = dataset_nori_index_dir

        