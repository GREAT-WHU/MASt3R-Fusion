#!/bin/bash



# for folder in 0000 0002 0003 0004 0005 0006 0009 0010; do
for folder in 0005; do
    GPU_ID=5
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID \
    python main_loop.py --h5_file data_${folder}.h5 \
     --config config/base_kitti360.yaml \
     --loop_output loop_${folder}.pkl

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU_ID \
    python main_global_optimization.py \
        --graph_path graph_${folder}.pkl \
        --loop_path loop_${folder}.pkl \
        --config config/base_kitti360.yaml \
        --calib_path config/intrinsics_kitti360.yaml \
        --imu_path /mnt/nas/Dataset/KITTI-360/2013_05_28_drive_${folder}_sync/imu.txt \
        --imu_dt -0.04 \
        --result_path result_post_${folder}.txt
done
