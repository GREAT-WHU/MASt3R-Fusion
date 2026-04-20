#!/bin/bash

seqs=("whu1" "whu2")

CONFIG_FILE="config/base_whu.yaml"
CALIB_FILE="config/intrinsics_whu.yaml"
DATASET_ROOT="/workspace/mast3r_fusion_open/whu_dataset"


for i in "${!seqs[@]}"
do
    seq="${seqs[$i]}"
    dataset_dir="${DATASET_ROOT}/${seq}"
    gnss_path="${dataset_dir}/rtk.txt"
    img_dir="${dataset_dir}/cam0"
    imu_path="${dataset_dir}/adis_imu.txt"
    GPU_ID=4

    python main_loop.py --config "$CONFIG_FILE" --h5_file data_${seq}.h5 --loop_output loop_${seq}.pkl

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID \
    python main_global_optimization.py \
        --graph_path graph_${seq}.pkl \
        --loop_path loop_${seq}.pkl \
        --config ${CONFIG_FILE} \
        --calib_path ${CALIB_FILE} \
        --imu_path ${imu_path} \
        --imu_dt 0.0 \
        --enable_gnss \
        --gnss_path ${gnss_path} \
        --result_path result_post_gnss_${seq}.txt
done
