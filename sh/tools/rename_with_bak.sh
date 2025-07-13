#!/bin/bash

# 检查参数数量
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <目录路径> <匹配字符串>"
    exit 1
fi

folder="$1"
pattern="$2"

# 检查目录是否存在
if [ ! -d "$folder" ]; then
    echo "错误：目录不存在：$folder"
    exit 1
fi

# 遍历目录下的所有文件（不包括子目录）
for file in "$folder"/*; do
    if [ -f "$file" ] && [[ "$(basename "$file")" == *"$pattern"* ]]; then
        mv "$file" "$file.bak"
        echo "重命名：$file -> $file.bak"
    fi
done

