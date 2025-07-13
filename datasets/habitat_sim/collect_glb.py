import os
import argparse
import shutil
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(description="move all matched files specified by a regular expression (not only .glb) to a given folder. Ensure that there are no files with the same name in the files to be moved.")
parser.add_argument("-r", "--reg", type=str, help="regular expression path")
parser.add_argument("-d", "--dst", type=str, help="target folder.")
parser.add_argument("--dry-run", action='store_true')
args = parser.parse_args()

files = sorted(glob(os.path.expanduser(args.reg)))

if not os.path.exists(args.dst):
    os.makedirs(args.dst, exist_ok=True)

for idx, f in tqdm(enumerate(files), total=len(files)):
    fname = os.path.basename(f)
    if args.dry_run:
        print(f"{f} -> {os.path.join(args.dst, fname)}")
        if idx > 5:
            break
    else:
        shutil.move(f, os.path.join(args.dst, fname))