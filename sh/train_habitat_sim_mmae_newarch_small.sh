#!/bin/bash
if [[ $# -eq 0 ]]; then
    echo "usage: train.sh config_path platform(local/cluster) gpu_type(4090/l20, only required when platform==cluster)"
    exit
fi

config_path=$1

batchsize_per_gpu=64
max_epoch=50
sche_epoch=100
warmup_epochs=5
keep_freq=5

num_workers=8
num_gpu=4
num_cpu=10
memory=160
datasets=habitat_release
data_dir=../data/habitat-sim-data
output_dir="exp/mmae_newarch-small-habitat_sim-mask_0.8"

mkdir -p $output_dir

echo "torchrun --nproc_per_node=$num_gpu pretrain_mmae_new_arch.py -cfg $config_path --dataset $datasets --output_dir $output_dir --data_dir $data_dir \
                 --batch_size $batchsize_per_gpu \
                 --epochs $sche_epoch --max_epoch $max_epoch --warmup_epochs $warmup_epochs --keep_freq $keep_freq --print_freq $keep_freq"

platform=$2
if [[ "$platform" == "cluster" ]]; then
    gpu_type=$3
    echo "using dpflow"
    link_config="datasets.link_config.link_config"
    link_name="train"
    tlaunch --gpu=$num_gpu --cpu=$num_cpu --memory=$memory --positive-tag=$gpu_type -- \
        torchrun --nproc_per_node=$num_gpu pretrain_mmae_new_arch.py -cfg $config_path --dataset $datasets --output_dir $output_dir --data_dir $data_dir \
                 --batch_size $batchsize_per_gpu --num_workers $num_workers \
                 --epochs $sche_epoch --max_epoch $max_epoch --warmup_epochs $warmup_epochs --keep_freq $keep_freq \
                 --link_config $link_config --link_name $link_name
else
    torchrun --nproc_per_node=$num_gpu pretrain_mmae_new_arch.py -cfg $config_path --dataset $datasets --output_dir $output_dir --data_dir $data_dir \
                --batch_size $batchsize_per_gpu --num_workers $num_workers \
                --epochs $sche_epoch --max_epoch $max_epoch --warmup_epochs $warmup_epochs --keep_freq $keep_freq
fi