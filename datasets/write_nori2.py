import argparse
import os
import megfile
import trun
import nori2
import pickle
from tqdm import tqdm
from concurrent.futures import as_completed

import dplink

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="s3://ljh-data/croco")
    parser.add_argument("-ds", "--datasets", type=str, help="training sets seperated by '+'. [habitat_release, ARKitScenes, MegaDepth, 3DStreetView, IndoorVL]")
    parser.add_argument("--volume", type=str, help="nori volume dir")
    parser.add_argument("--index", type=str, help="nori index dir")
    parser.add_argument("--pairs_per_run", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--memory", type=int, default=4, help="in GB")
    parser.add_argument("--log_dir", type=str, default="nori_logs")
    args = parser.parse_args()
    return args

def get_trun_spec(num_workers, memory, log_dir):
    spec = trun.RunnerSpec()
    spec.resources.cpu = 1
    spec.resources.memory=memory
    spec.resources.gpu=0
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        spec.log_dir = log_dir
    return spec

def upload_nori(data_root, dsname, pairs:list, volume_dir:str):
    volume_path = os.path.join(volume_dir, f"{dsname}.nori")
    nw = nori2.remotewriteopen(volume_path)
    nori_index = {}
    for path1, path2 in pairs:
        with megfile.smart_open(path1, 'rb') as f:
            img1 = f.read()
        with megfile.smart_open(path2, 'rb') as f:
            img2 = f.read()
        relpath1 = os.path.relpath(path1, data_root)
        relpath2 = os.path.relpath(path2, data_root)
        basepath = "_".join(relpath1.split("_")[:-1])

        nid1 = nw.put(img1, filename=relpath1)
        nid2 = nw.put(img2, filename=relpath2)
        pair_index = {
            'img_1': nid1,
            'img_2': nid2
        }

        if dsname == 'habitat_release':  # depth file exists.
            path_d_1 = os.path.splitext(path1)[0] + "_depth.png"
            path_d_2 = os.path.splitext(path2)[0] + "_depth.png"
            with megfile.smart_open(path_d_1, 'rb') as f:
                dep1 = f.read()
            with megfile.smart_open(path_d_2, 'rb') as f:
                dep2 = f.read()
            relpath_d_1 = os.path.relpath(path_d_1, data_root)
            relpath_d_2 = os.path.relpath(path_d_2, data_root)
            nid_d_1 = nw.put(dep1, filename=relpath_d_1)
            nid_d_2 = nw.put(dep2, filename=relpath_d_2)
            pair_index['depth_1'] = nid_d_1
            pair_index['depth_2'] = nid_d_2
        
        nori_index[basepath] = pair_index
    nw.close()
    return nori_index

if __name__ == "__main__":
    from .utils import dsname_to_image_pairs

    args = parse_args()
    data_root = args.data_root
    datasets = args.datasets
    datasets = datasets.split("+")
    volume_dir = args.volume
    index_dir = args.index
    pairs_per_run = args.pairs_per_run
    ds_pairs = {}
    for dsname in datasets:
        pairs = dsname_to_image_pairs(dsname, data_root)
        ds_pairs[dsname] = pairs
        print(f"{dsname}: {len(pairs)} pairs.")

    for dsname, pairs in ds_pairs.items():
        spec = get_trun_spec(args.num_workers, args.memory, args.log_dir)
        print(f"process {dsname}")
        num_pairs = len(pairs)
        nori_idx = {}
        with trun.TRunExecutor(spec, args.num_workers) as executer:
            cnt = 0
            futures = []
            while cnt < num_pairs:
                p = pairs[cnt : min(cnt + pairs_per_run, num_pairs)]
                fu = executer.submit(
                    upload_nori, data_root, dsname, p, volume_dir
                )
                futures.append(fu)
                cnt += pairs_per_run

            for fu in tqdm(as_completed(futures), total=len(futures)):
                ret = fu.result()
                nori_idx.update(ret)

        nori_idx_path = os.path.join(index_dir, f"{dsname}.idx")
        print(f"write nori index to {nori_idx_path}")
        with megfile.smart_open(nori_idx_path, 'wb') as f:
            pickle.dump(nori_idx, f)