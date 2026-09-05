#!/usr/bin/env bash
set -euo pipefail

dataset_root="${1:-datasets/euroc}"
gpu_id="${GPU_ID:-0}"
config_file="config/base_euroc.yaml"
calib_file="config/intrinsics_euroc.yaml"

if (( $# > 1 )); then
    sequences=("${@:2}")
else
    sequences=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult)
fi

mkdir -p results/euroc

for sequence in "${sequences[@]}"; do
    sequence_path="${dataset_root}/${sequence}"
    if [[ ! -f "${sequence_path}/mav0/cam0/data.csv" || ! -f "${sequence_path}/mav0/imu0/data.csv" ]]; then
        echo "Missing EuRoC sequence or MAV files: ${sequence_path}" >&2
        exit 1
    fi

    echo "Running ${sequence} on GPU ${gpu_id}"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="${gpu_id}" python main.py \
        --dataset "${sequence_path}" \
        --config "${config_file}" \
        --calib "${calib_file}" \
        --imu_path "${sequence_path}/mav0/imu0/data.csv" \
        --imu_dt 0.0 \
        --result_path "results/euroc/result_${sequence}.txt" \
        --save_h5 \
        --no-viz

    mv graph.pkl "results/euroc/graph_${sequence}.pkl"
    mv data.h5 "results/euroc/data_${sequence}.h5"
done
