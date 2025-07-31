#!/bin/bash
if [[ $# -eq 0 ]]; then
    echo "usage: train_habitat_sim_mmae.sh model_size(small/large) pose_emb_type(cosine/curope) platform(local/cluster) gpu_type(4090/l20, only required when platform==cluster)"
    exit
fi

model_size=$1
pose_emb_type=$2
batchsize_per_gpu=64  # 更小的batch?
max_epoch=50
sche_epoch=100
warmup_epochs=5
keep_freq=5
mask_ratio=0.8
num_dataloader_workers=4

num_gpu=4
num_cpu=8
memory=128
datasets=habitat_release
data_dir=../data/habitat-sim-data
output_dir="exp/mmae-habitat_sim-CroCo_${model_size}-${pose_emb_type}_mask-0.8"

if [[ "$model_size" == small ]]; then
    model="MMAE_CroCoNet(mask_ratio=${mask_ratio}"
elif [[ "$model_size" == large ]]; then
    model="MMAE_CroCoNet(mask_ratio=${mask_ratio}, enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_num_heads=12, dec_depth=12"
else
    echo "Not implemented model size: $model"
    exit
fi

if [[ "$pose_emb_type" == "curope" ]]; then
    if [[ "$model" == *"(" ]]; then
        model="${model}pos_embed='RoPE100'"
    else
        model="$model, pos_embed='RoPE100'"
    fi
elif [[ "$pose_emb_type" != "cosine" ]]; then
    echo "Not implemented pose_emb_type: $pose_emb_type"
    exit
fi

model="$model)"

mkdir -p $output_dir

echo "torchrun --nproc_per_node=$num_gpu pretrain_mmae.py --model "\"$model\"" --dataset $datasets --output_dir $output_dir --data_dir $data_dir \
                 --batch_size $batchsize_per_gpu \
                 --epochs $sche_epoch --max_epoch $max_epoch --warmup_epochs $warmup_epochs --num_workers $num_dataloader_workers"

platform=$3
if [[ "$platform" == "cluster" ]]; then
    gpu_type=$4
    echo "using dpflow"
    link_config="datasets.link_config.link_config"
    link_name="train"
    tlaunch --gpu=$num_gpu --cpu=$num_cpu --memory=$memory --positive-tag=$gpu_type -- \
        torchrun --nproc_per_node=$num_gpu pretrain_mmae.py --model "\"$model\"" --dataset $datasets --output_dir $output_dir --data_dir $data_dir \
                 --batch_size $batchsize_per_gpu \
                 --epochs $sche_epoch --max_epoch $max_epoch --warmup_epochs $warmup_epochs --keep_freq $keep_freq \
                 --link_config $link_config --link_name $link_name --num_workers $num_dataloader_workers
else
    torchrun --nproc_per_node=$num_gpu pretrain_mmae.py --model "\"$model\"" --dataset $datasets --output_dir $output_dir --data_dir $data_dir \
                --batch_size $batchsize_per_gpu \
                --epochs $sche_epoch --max_epoch $max_epoch --warmup_epochs $warmup_epochs --keep_freq $keep_freq --num_workers $num_dataloader_workers
fi