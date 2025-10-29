#!/bin/bash


# for folder in handheld1 handheld2 overexposure; do
for folder in overexposure; do
    GPU_ID=5

    [ "$folder" = 'handheld1' ]    && imu_path=/mnt/nas/Dataset/SubT_MRS/Handheld1_Folder/imu/imu_data.csv
    [ "$folder" = 'handheld2' ]    && imu_path=/mnt/nas/Dataset/SubT_MRS/Handheld2_Folder/imu/imu_data.csv
    [ "$folder" = 'overexposure' ] && imu_path=/mnt/nas/Dataset/SubT_MRS/OverExposure_Folder/imu/imu_data.csv

    [ "$folder" = 'handheld1' ]    && calib_path=config/intrinsics_subt_handheld.yaml
    [ "$folder" = 'handheld2' ]    && calib_path=config/intrinsics_subt_handheld.yaml
    [ "$folder" = 'overexposure' ] && calib_path=config/intrinsics_subt_overexposure.yaml

    [ "$folder" = 'handheld1' ]    && config_file=config/base_subt_handheld.yaml
    [ "$folder" = 'handheld2' ]    && config_file=config/base_subt_handheld.yaml
    [ "$folder" = 'overexposure' ] && config_file=config/base_subt_overexposure.yaml

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID \
    python main_loop.py --config "$config_file" --h5_file data_${folder}.h5 --loop_output loop_${folder}.pkl

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID \
    python main_global_optimization.py \
        --graph_path graph_${folder}.pkl \
        --loop_path loop_${folder}.pkl \
        --calib_path ${calib_path} \
        --config ${config_file} \
        --imu_path ${imu_path} \
        --imu_dt 0.0 \
        --result_path result_post_${folder}.txt
done