import os
import subprocess
import argparse
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser(description="convert all .ply files specified by a regular expression to .glb files. The Assimp command-line tool must be installed.")
parser.add_argument("-i", "--src", type=str, help="A regular expression specifying .ply files to be converted.")
parser.add_argument("-rs", "--remove-suffix", type=str, default="", help="the suffix to be removed from the filename.")
parser.add_argument("-d", "--delete", action="store_true", help="delete the origin .ply file after conversion.")
args = parser.parse_args()
len_suffix = len(args.remove_suffix)
if args.src.startswith("~"):
    args.src = os.path.expanduser(args.src)
plys = sorted(glob(args.src))
for plypath in tqdm(plys):
    name = os.path.basename(plypath)
    basename, ext = os.path.splitext(name)
    if basename.endswith(args.remove_suffix):
        basename = basename[:-len_suffix]
    glbpath = os.path.join(os.path.dirname(plypath), basename + ".glb")
    cmd = ['assimp', 'export', plypath, glbpath]
    ret = subprocess.run(cmd, stdout=subprocess.PIPE)
    if ret.returncode != 0:
        print("Conversion failed: ", " ".join(cmd))
    else:
        if args.delete:
            os.remove(plypath)