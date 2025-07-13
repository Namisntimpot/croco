# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""
Script generating commandlines to generate image pairs from metadata files.
"""
import os
import glob
from tqdm import tqdm
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log", default=None, help="used to record failure cases.")
    args = parser.parse_args()

    log = None
    if args.log is not None:
        log = open(args.log, 'w')

    count = sum([1 for _ in glob.iglob(f"{args.input_dir}/**/metadata.json", recursive=True)])
    print(f"metadata files count: {count}")
    input_metadata_filenames = glob.iglob(f"{args.input_dir}/**/metadata.json", recursive=True)

    for metadata_filename in tqdm(input_metadata_filenames, total=count):
        output_dir = os.path.join(args.output_dir, os.path.relpath(os.path.dirname(metadata_filename), args.input_dir))
        # Do not process the scene if the metadata file already exists
        if os.path.exists(os.path.join(output_dir, "metadata.json")):
            continue
        cmd = ['python', '-m', 'datasets.habitat_sim.generate_from_metadata', '--metadata_filename', metadata_filename, '--output_dir', output_dir]
        ret = subprocess.run(cmd, stdout=subprocess.PIPE)
        if ret.returncode != 0:
            failinfo = f"Fail: {' '.join(cmd)}"
            print(failinfo)
            if log is not None:
                log.write(failinfo+"\n")
    
    if log is not None:
        log.close()