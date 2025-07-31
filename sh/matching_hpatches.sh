#!/bin/bash

if [[ $# -eq 0 ]]; then
    echo "usage: matching_hpatches_croco.sh config_path hpatch_path output_path [args of matching_task.py]"
    exit
fi

config_path=$1
hpatches_path=$2
output_path=$3

shift 3

COMMAND_NAME=tlaunch
if command -v "$COMMAND_NAME" &> /dev/null; then
    gpu=1
    cpu=4
    memory=32
    gputype=4090
    tlaunch --gpu=$gpu --cpu=$cpu --memory=$memory --positive-tag=$gputype -- \
        python matching_task.py -cfg $config_path --hpatches_root $hpatches_path \
        -o $output_path -o_attn $@
else
    python matching_task.py -cfg $config_path --hpatches_root $hpatches_path \
        -o $output_path -o_attn $@
fi