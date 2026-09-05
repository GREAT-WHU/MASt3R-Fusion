#!/usr/bin/env python3
"""Evaluate MASt3R-Fusion trajectories on EuRoC with evo."""

import argparse
import copy
import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EUROC_SEQUENCES = (
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
)


def _trajectory(positions, quaternions_xyzw, timestamps):
    quaternions_xyzw = np.asarray(quaternions_xyzw, dtype=np.float64)
    norms = np.linalg.norm(quaternions_xyzw, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("trajectory contains an invalid zero quaternion")
    quaternions_xyzw /= norms[:, None]
    return PoseTrajectory3D(
        positions_xyz=np.asarray(positions, dtype=np.float64),
        orientations_quat_wxyz=quaternions_xyzw[:, [3, 0, 1, 2]],
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def load_euroc_groundtruth(path, body_T_camera):
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.shape[1] < 8:
        raise ValueError(f"invalid EuRoC ground truth: {path}")

    timestamps = data[:, 0] / 1e9
    positions_body = data[:, 1:4]
    rotations_world_body = Rotation.from_quat(data[:, [5, 6, 7, 4]]).as_matrix()

    rotation_body_camera = body_T_camera[:3, :3]
    translation_body_camera = body_T_camera[:3, 3]
    positions_camera = positions_body + np.einsum(
        "nij,j->ni", rotations_world_body, translation_body_camera
    )
    rotations_world_camera = rotations_world_body @ rotation_body_camera
    quaternions_camera = Rotation.from_matrix(rotations_world_camera).as_quat()
    return _trajectory(positions_camera, quaternions_camera, timestamps)


def load_estimate(path, include_pre_init=False, init_keyframes=7):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 8:
        raise ValueError(f"invalid estimate trajectory: {path}")

    timestamps = data[:, 0].copy()
    timestamps[timestamps > 1e12] /= 1e9
    data[:, 0] = timestamps

    start_frame = None
    if not include_pre_init and data.shape[1] >= 17:
        frame_ids = data[:, 15].astype(np.int64)
        keyframe_ids = np.unique(frame_ids[data[:, 16] > 0.5])
        if keyframe_ids.size >= init_keyframes:
            start_frame = int(keyframe_ids[init_keyframes - 1])
            data = data[frame_ids >= start_frame]

    # VI initialization rewrites the initial keyframes. Keep the last estimate
    # for each timestamp, then restore chronological order for evo.
    last_by_timestamp = {float(row[0]): row for row in data}
    data = np.asarray([last_by_timestamp[t] for t in sorted(last_by_timestamp)])
    estimate = _trajectory(data[:, 1:4], data[:, 4:8], data[:, 0])
    return estimate, start_frame


def save_plot(path, reference, estimate, sequence):
    figure = plt.figure(figsize=(6, 5))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*reference.positions_xyz.T, "k--", linewidth=1.0, label="ground truth")
    axis.plot(*estimate.positions_xyz.T, linewidth=1.0, label="MASt3R-Fusion")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(f"EuRoC {sequence} — SE(3)-aligned APE")
    axis.legend()

    extent = np.ptp(np.vstack((reference.positions_xyz, estimate.positions_xyz)), axis=0)
    axis.set_box_aspect(np.maximum(extent, 1e-6))
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_sequence(sequence, dataset_root, result_path, output_dir, body_T_camera,
                      max_time_diff, correct_scale, include_pre_init):
    groundtruth_path = (
        dataset_root / sequence / "mav0/state_groundtruth_estimate0/data.csv"
    )
    if not groundtruth_path.is_file():
        raise FileNotFoundError(f"ground truth not found: {groundtruth_path}")
    if not result_path.is_file():
        raise FileNotFoundError(f"estimate not found: {result_path}")

    reference = load_euroc_groundtruth(groundtruth_path, body_T_camera)
    estimate, start_frame = load_estimate(
        result_path, include_pre_init=include_pre_init
    )
    reference, estimate = sync.associate_trajectories(
        reference,
        estimate,
        max_diff=max_time_diff,
        first_name="EuRoC ground truth",
        snd_name="MASt3R-Fusion",
    )
    if reference.num_poses < 2:
        raise ValueError(f"too few associated poses for {sequence}")

    estimate_aligned = copy.deepcopy(estimate)
    estimate_aligned.align(reference, correct_scale=correct_scale)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((reference, estimate_aligned))
    statistics = ape.get_all_statistics()
    result = ape.get_result("EuRoC ground truth", "MASt3R-Fusion")
    result.info["title"] = "APE translation (SE(3) aligned)"
    result.add_trajectory("EuRoC ground truth", reference)
    result.add_trajectory("MASt3R-Fusion", estimate_aligned)
    result.add_np_array("timestamps", estimate_aligned.timestamps)

    sequence_output = output_dir / sequence
    sequence_output.mkdir(parents=True, exist_ok=True)
    file_interface.write_tum_trajectory_file(
        sequence_output / "groundtruth_camera.tum", reference
    )
    file_interface.write_tum_trajectory_file(
        sequence_output / "estimate_aligned.tum", estimate_aligned
    )
    file_interface.save_res_file(sequence_output / "ape.zip", result)
    save_plot(sequence_output / "ape.png", reference, estimate_aligned, sequence)

    return {
        "sequence": sequence,
        "poses": reference.num_poses,
        "start_frame": "" if start_frame is None else start_frame,
        **{name: statistics[name] for name in ("rmse", "mean", "median", "std", "min", "max")},
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MASt3R-Fusion EuRoC trajectories with evo APE."
    )
    parser.add_argument("sequences", nargs="*", metavar="SEQUENCE")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/euroc"))
    parser.add_argument("--results-root", type=Path, default=Path("results/euroc"))
    parser.add_argument("--calib", type=Path, default=Path("config/intrinsics_euroc.yaml"))
    parser.add_argument("--result-pattern", default="result_{sequence}.txt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-time-diff", type=float, default=0.02)
    parser.add_argument("--correct-scale", action="store_true")
    parser.add_argument("--include-pre-init", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    unknown_sequences = sorted(set(args.sequences) - set(EUROC_SEQUENCES))
    if unknown_sequences:
        raise ValueError(
            "unknown EuRoC sequence(s): " + ", ".join(unknown_sequences)
        )
    sequences = args.sequences or EUROC_SEQUENCES
    output_dir = args.output_dir or args.results_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.calib.open() as calibration_file:
        body_T_camera = np.asarray(
            yaml.safe_load(calibration_file)["Tic"], dtype=np.float64
        )

    rows = []
    for sequence in sequences:
        result_path = args.results_root / args.result_pattern.format(sequence=sequence)
        row = evaluate_sequence(
            sequence,
            args.dataset_root,
            result_path,
            output_dir,
            body_T_camera,
            args.max_time_diff,
            args.correct_scale,
            args.include_pre_init,
        )
        rows.append(row)
        print(
            f"{sequence}: poses={row['poses']} "
            f"APE RMSE={row['rmse']:.6f} m median={row['median']:.6f} m"
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
