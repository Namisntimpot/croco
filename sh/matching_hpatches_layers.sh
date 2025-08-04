#!/bin/bash

if [[ $# -eq 0 ]]; then
    echo "usage: matching_hpatches_croco.sh config_path hpatch_path output_root num_dec_layers(8 for small, 12 for large) [args of matching_task.py]"
    exit
fi

config_path=$1
hpatches_path=$2
output_root=$3
num_dec_layers=$4

shift 4

for i in $(seq 0 $((num_dec_layers-1)))
do
    echo "Decoder layer: $i"
    if [[ $output_root == */ ]]; then
        output_path="${output_root}${i}"
    else
        output_path="${output_root}/${i}"
    fi
    echo "Output path: ${output_path}"

    COMMAND_NAME=tlaunch
    if command -v "$COMMAND_NAME" &> /dev/null; then
        gpu=1
        cpu=4
        memory=32
        gputype=4090
        tlaunch --gpu=$gpu --cpu=$cpu --memory=$memory --positive-tag=$gputype -- \
            python matching_task.py -cfg $config_path --hpatches_root $hpatches_path \
            -o $output_path -o_attn $@ --attn_layers_adopted $i
    else
        python matching_task.py -cfg $config_path --hpatches_root $hpatches_path \
            -o $output_path -o_attn $@ --attn_layers_adopted $i
    fi
done