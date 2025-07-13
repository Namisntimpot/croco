#!/bin/bash

for libfile in $(ls /lib/x86_64-linux-gnu/*.535.154.05); do
    # 从完整文件名中提取基础链接名，例如从 libnvidia-ml.so.535.154.05 提取 libnvidia-ml.so.1
    # 这通常是通过去掉最后的两个版本号部分来获得，但更稳妥的方式是直接找对应的 .so.1 文件
    base_link_name=$(echo "$libfile" | sed -E 's/\.so\.[0-9]+\.[0-9]+\.[0-9]+$/\.so\.1/')
    # 注意：还会有一个漏网之鱼：libnvidia-nvvm.so.4，它没有.so.1

    # 检查对应的 .so.1 链接是否存在
    if [ -L "$base_link_name" ]; then
        # 获取目标文件名，例如 libnvidia-ml.so.535.154.05
        target_file_name=$(basename "$libfile")
        
        echo "Fixing link for $base_link_name -> $target_file_name"
        
        # 强制删除旧的错误链接，并创建指向正确版本的新链接
        sudo ln -sf "$target_file_name" "$base_link_name"
    fi
done

echo "Symbolic link correction finished."
echo "注意还有一个漏网之鱼需要手动处理：libnvidia-nvvm.so.4"
