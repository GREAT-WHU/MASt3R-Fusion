#!/bin/bash


base_dataset_path="/mnt/nas/Dataset/KITTI-360"
config_file="config/base_kitti360.yaml"
calib_file="config/intrinsics_kitti360.yaml"
imu_dt="-0.04"

# for folder in 0000 0002 0003 0004 0005 0006 0009 0010; do
for folder in 0005; do
    GPU_ID=5
    echo "Using GPU $GPU_ID for sequence $folder"

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
        --dataset "${base_dataset_path}/2013_05_28_drive_${folder}_sync/image_00/data_rgb" \
        --config "$config_file" \
        --calib "$calib_file" \
        --imu_path "${base_dataset_path}/2013_05_28_drive_${folder}_sync/imu.txt" \
        --imu_dt "$imu_dt" \
        --stamp_path "${base_dataset_path}/2013_05_28_drive_${folder}_sync/camstamp.txt" \
        --result_path "result_${folder}.txt" \
        --save_h5
        #--no-viz   # uncomment this for headless mode
    mv graph.pkl graph_${folder}.pkl
    mv data.h5 data_${folder}.h5
done
