#!/bin/bash

set -e

GPU_ID=1
CONFIG_FILE="config/base_whu.yaml"
CALIB_FILE="config/intrinsics_whu.yaml"
IMU_DT="-0.00"

DATASET_ROOT="/workspace/mast3r_fusion_open/whu_dataset"

seqs=("whu1" "whu2")

for i in "${!seqs[@]}"
do
    seq="${seqs[$i]}"

    dataset_dir="${DATASET_ROOT}/${seq}"
    img_dir="${dataset_dir}/cam0"
    imu_path="${dataset_dir}/adis_imu.txt"
    stamp_path="${dataset_dir}/stamp.txt"

    result_file="result_${seq}.txt"

    echo "=============================="
    echo "Running sequence: $seq"
    echo "Using GPU: $GPU_ID"
    echo "=============================="

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
        --dataset "$img_dir" \
        --config "$CONFIG_FILE" \
        --calib "$CALIB_FILE" \
        --imu_path "$imu_path" \
        --imu_dt "$IMU_DT" \
        --stamp_path "$stamp_path" \
        --result_path "$result_file" \
        --save_h5 \
        --no-viz 

    mv -f graph.pkl "graph_${seq}.pkl"
    mv -f data.h5 "data_${seq}.h5"

done