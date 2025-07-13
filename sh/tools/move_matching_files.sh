#!/bin/bash

# 检查参数数量
if [ "$#" -ne 3 ]; then
    echo "用法: $0 <源文件夹> <匹配字符串> <目标临时文件夹>"
    exit 1
fi

src_dir="$1"
match_str="$2"
tmp_dir="$3"

# 检查源目录是否存在
if [ ! -d "$src_dir" ]; then
    echo "错误：源目录不存在：$src_dir"
    exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p "$tmp_dir"

# 遍历并移动匹配的文件（不包括子目录）
for file in "$src_dir"/*; do
    if [ -f "$file" ] && [[ "$(basename "$file")" == *"$match_str"* ]]; then
        mv "$file" "$tmp_dir/"
        echo "已移动：$(basename "$file") 到 $tmp_dir"
    fi
done

