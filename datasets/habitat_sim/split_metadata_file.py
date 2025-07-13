import os
import shutil
import argparse

def split_and_copy_folders(src_folder, dst_folder1, dst_folder2):
    # 获取 A 文件夹中的子文件夹（只包括目录，不包括文件）
    subfolders = [f for f in sorted(os.listdir(src_folder)) 
                  if os.path.isdir(os.path.join(src_folder, f))]

    total = len(subfolders)
    half = total // 2

    # 前一半 -> B，后一半 -> C
    first_half = subfolders[:half]
    second_half = subfolders[half:]

    # 创建目标文件夹（如果不存在）
    os.makedirs(dst_folder1, exist_ok=True)
    os.makedirs(dst_folder2, exist_ok=True)

    print(f"总共 {total} 个子文件夹，前一半 ({len(first_half)}) 复制到 {dst_folder1}，后一半 ({len(second_half)}) 复制到 {dst_folder2}")

    # 复制前一半
    for folder in first_half:
        src = os.path.join(src_folder, folder)
        dst = os.path.join(dst_folder1, folder)
        shutil.copytree(src, dst)
    
    # 复制后一半
    for folder in second_half:
        src = os.path.join(src_folder, folder)
        dst = os.path.join(dst_folder2, folder)
        shutil.copytree(src, dst)

    print("复制完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将A中的子文件夹一半复制到B，一半复制到C")
    parser.add_argument("A", help="源文件夹A")
    parser.add_argument("B", help="目标文件夹B（前一半）")
    parser.add_argument("C", help="目标文件夹C（后一半）")
    args = parser.parse_args()

    split_and_copy_folders(args.A, args.B, args.C)
