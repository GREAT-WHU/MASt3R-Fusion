#!/bin/bash



# config_file="config/base_subt_handheld.yaml"
# calib_file="config/intrinsics_subt_handheld.yaml"
# imu_dt="-0.00"
# GPU_ID=5
# echo "Using GPU $GPU_ID for sequence"
# DIR="Handheld1_Folder"
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
#     --dataset "/mnt/nas/Dataset/SubT_MRS/${DIR}/cam_0" \
#     --config "$config_file" \
#     --calib "$calib_file" \
#     --imu_path "/mnt/nas/Dataset/SubT_MRS/${DIR}/imu/imu_data.csv" \
#     --imu_dt "$imu_dt" \
#     --stamp_path "/mnt/nas/Dataset/SubT_MRS/${DIR}/cam_0/timestamps.txt" \
#     --start_from 2300 \
#     --result_path "result_handheld1.txt" \
#     --save_h5
#     #--no-viz   # uncomment this for headless mode
# mv graph.pkl graph_handheld1.pkl
# mv data.h5 data_handheld1.h5

# DIR="Handheld2_Folder"
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
#     --dataset "/mnt/nas/Dataset/SubT_MRS/${DIR}/cam_0" \
#     --config "$config_file" \
#     --calib "$calib_file" \
#     --imu_path "/mnt/nas/Dataset/SubT_MRS/${DIR}/imu/imu_data.csv" \
#     --imu_dt "$imu_dt" \
#     --stamp_path "/mnt/nas/Dataset/SubT_MRS/${DIR}/cam_0/timestamps.txt" \
#     --start_from 2300 \
#     --result_path "result_handheld2.txt" \
#     --save_h5
#     #--no-viz   # uncomment this for headless mode
# mv graph.pkl graph_handheld2.pkl
# mv data.h5 data_handheld2.h5

config_file="config/base_subt_overexposure.yaml"
calib_file="config/intrinsics_subt_overexposure.yaml"
imu_dt="-0.00"
GPU_ID=5
echo "Using GPU $GPU_ID for sequence"
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --dataset "/mnt/nas/Dataset/SubT_MRS/OverExposure_Folder/cam_0" \
    --config "$config_file" \
    --calib "$calib_file" \
    --imu_path "/mnt/nas/Dataset/SubT_MRS/OverExposure_Folder/imu/imu_data.csv" \
    --imu_dt "$imu_dt" \
    --stamp_path "/mnt/nas/Dataset/SubT_MRS/OverExposure_Folder/cam_0/timestamps.txt" \
    --start_from 600 \
    --result_path "result_overexposure.txt" \
    --save_h5
    #--no-viz   # uncomment this for headless mode
mv graph.pkl graph_overexposure.pkl
mv data.h5 data_overexposure.h5

