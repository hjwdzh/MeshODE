#!/bin/bash

run_name=run_0
log_dir=logs/$run_name
data_root=../../data/shapenet

export CUDA_VISIBLE_DEVICES=2

python train_shapenet.py \
--batch_size_per_gpu=4 \
--epochs=100 \
--lr=1e-3 \
--log_dir=$log_dir \
--nsamples=2048 \
--lat_dims=64 \
--encoder_nf=32 \
--deformer_nf=64 \
--lr_scheduler \
--no_normals \
--visualize_mesh \