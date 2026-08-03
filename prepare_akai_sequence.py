import cv2
import json
import argparse
import pathlib
import os
import numpy as np

DATA_ROOT = pathlib.Path("data").resolve()


def _default_output_dir(raw_dir: pathlib.Path) -> pathlib.Path:
    """Mirror a raw chunk's path below data/ inside data/normalized_data/."""
    try:
        relative_raw_dir = raw_dir.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Raw input must be below {DATA_ROOT} when --output is omitted. "
            "Pass --output explicitly for an external raw directory."
        ) from exc
    return DATA_ROOT / "normalized_data" / relative_raw_dir


def _find_capture_files(raw_dir: pathlib.Path, chunk: str | None):
    """Return the left-video and IMU files stored directly in an Akai chunk."""
    if chunk is not None:
        video_path = raw_dir / f"left_{chunk}.mjpeg"
        imu_path = raw_dir / f"imu_{chunk}.jsonl"
        missing = [path for path in (video_path, imu_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Could not find the requested Akai files:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )
        return video_path, imu_path

    videos = sorted(raw_dir.glob("left_*.mjpeg"))
    if len(videos) != 1:
        found = ", ".join(path.name for path in videos) or "none"
        raise ValueError(
            f"Expected exactly one left_*.mjpeg in {raw_dir}; found {found}. "
            "Pass --chunk when the folder contains multiple chunks."
        )

    chunk = videos[0].stem.removeprefix("left_")
    imu_path = raw_dir / f"imu_{chunk}.jsonl"
    if not imu_path.is_file():
        raise FileNotFoundError(f"Missing IMU file: {imu_path}")
    return videos[0], imu_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert one downloaded Akai raw chunk to MASt3R-Fusion input.",
        epilog=(
            "Example: python prepare_akai_sequence.py "
            "--input data/stage-humyn-egocentric-stereo-data/raw/akai/akai-001/"
            "akai-ego-001_2026-07-16_22-18-16/005"
        ),
    )
    parser.add_argument(
        '--input', type=pathlib.Path, required=True, metavar='RAW_CHUNK_DIR',
        help='Downloaded Akai chunk directory containing calibration.json, left_*.mjpeg, and imu_*.jsonl.',
    )
    parser.add_argument(
        '--chunk', type=str, default=None,
        help='Optional chunk identifier (for example 005); inferred from left_*.mjpeg when omitted.',
    )
    parser.add_argument(
        '--output', type=pathlib.Path, default=None, metavar='NORMALIZED_DIR',
        help='Normalized output directory. By default, mirrors the raw path below data/normalized_data/.',
    )
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    input_dir = args.input.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Raw Akai chunk directory does not exist: {input_dir}")

    video_path, imu_jsonl = _find_capture_files(input_dir, args.chunk)
    calibration_path = input_dir / "calibration.json"
    if not calibration_path.is_file():
        parser.error(f"Missing calibration file: {calibration_path}")

    try:
        output_dir = (
            args.output.expanduser()
            if args.output is not None
            else _default_output_dir(input_dir)
        )
    except ValueError as exc:
        parser.error(str(exc))
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Normalized output: {output_dir.resolve()}")

    # 1. Load calibration
    with open(calibration_path, "r") as f:
        calib = json.load(f)
    
    cam0 = calib['cameras']['cam0']
    W, H = cam0['resolution']
    K = np.eye(3)
    K[0, 0] = cam0['intrinsics'][0]
    K[1, 1] = cam0['intrinsics'][1]
    K[0, 2] = cam0['intrinsics'][2]
    K[1, 2] = cam0['intrinsics'][3]
    D = np.array(cam0['distortion_coefficients'])

    # Optimal new camera matrix
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2)
    
    timeshift_cam_imu_s = calib.get('temporal', {}).get('timeshift_cam_imu_s', 0.0)

    # 2. Extract IMU and frame timestamps
    imu_records = []
    frame_times = []
    
    with open(imu_jsonl, "r") as f:
        for line in f:
            record = json.loads(line)
            t_us = record['t_us']
            
            # IMU conversion
            ax = record['ax'] / 16384.0 * 9.80665
            ay = record['ay'] / 16384.0 * 9.80665
            az = record['az'] / 16384.0 * 9.80665
            gx = record['gx'] / 16.4 * (np.pi / 180.0)
            gy = record['gy'] / 16.4 * (np.pi / 180.0)
            gz = record['gz'] / 16.4 * (np.pi / 180.0)
            t_sec = t_us / 1e6
            
            imu_records.append([t_sec, gx, gy, gz, ax, ay, az])
            
            if record.get('fsync_flag', 0) == 1:
                # camera timestamp based on sync
                t_cam_sec = (t_us - record.get('fsync_delay_us', 0)) / 1e6
                t_cam_sec -= timeshift_cam_imu_s
                frame_times.append(t_cam_sec)
    
    imu_records = np.array(imu_records)
    
    # 3. Trim frames
    # The last 3 frames are trimmed
    num_frames = len(frame_times) - 3
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
        
    frame_times = frame_times[:num_frames]
    
    # 4. Save camera timestamps
    stamp_path = output_dir / "camera_timestamps.txt"
    with open(stamp_path, "w") as f:
        for t in frame_times:
            f.write(f"{t:.6f}\n")
            
    # 5. Save IMU
    imu_path = output_dir / "imu.csv"
    np.savetxt(imu_path, imu_records, delimiter=",", fmt="%.6f")
    
    # 6. Save camera.yaml
    camera_yaml = output_dir / "camera.yaml"
    T_cam0_imu = calib['extrinsics']['T_cam0_imu']
    with open(camera_yaml, "w") as f:
        f.write(f"width: {W}\n")
        f.write(f"height: {H}\n")
        f.write(f"calibration: [{K_new[0,0]:.6f}, {K_new[1,1]:.6f}, {K_new[0,2]:.6f}, {K_new[1,2]:.6f}]\n")
        f.write("Tic:\n")
        for row in T_cam0_imu:
            f.write(f"  - [{row[0]:.8f},{row[1]:.8f},{row[2]:.8f},{row[3]:.8f}]\n")
            
    # 7. Extract and rectify frames
    print("Extracting and rectifying frames...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open MJPEG video: {video_path}")
    frame_idx = 0
    while cap.isOpened() and frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Rectify
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Retain the established normalized-data JPEG layout.
        out_path = frames_dir / f"{frame_times[frame_idx]:.6f}.jpg"
        cv2.imwrite(str(out_path), undistorted)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{num_frames} frames...")
    
    cap.release()
    print("Done!")
    
    # Save mast3r_akai.yaml config to reference base_whu.yaml
    config_yaml = output_dir / "mast3r_akai.yaml"
    with open(config_yaml, "w") as f:
        f.write('inherit: "config/base_whu.yaml"\n')
        f.write('dataset:\n')
        f.write('  center_principle_point: False\n')
        f.write('  subsample: 1\n')
        f.write('  img_downsample: 1\n')

if __name__ == "__main__":
    main()
