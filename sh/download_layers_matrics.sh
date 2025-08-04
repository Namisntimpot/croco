#!/bin/bash

if [[ $# -eq 0 ]]; then
    echo "usage: download_layers_matrics.sh result_root output_root num_dec_layers(8 for small, 12 for large)"
    exit
fi

result_root=$1
output_root=$2
num_dec_layers=$3


for i in $(seq 0 $((num_dec_layers-1)))
do
    echo "Decoder layer: $i"
    if [[ $result_root == */ ]]; then
        result_path="${result_root}${i}/matrics.json"
    else
        result_path="${result_root}/${i}/matrics.json"
    fi
    echo "Result path: ${result_path}"
    if [[ $output_root == */ ]]; then
        output_path="${output_root}${i}.json"
    else
        output_path="${output_root}/${i}.json"
    fi
    echo "Output path: ${output_path}"
    megfile cp $result_path $output_path
done