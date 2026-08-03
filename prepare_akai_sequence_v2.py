"""Prepare a normalized Akai capture containing left_rectified.mp4.

This is the rectified counterpart of prepare_akai_sequence.py.  IMU conversion,
fsync timestamps, and rectification-skipping are unchanged; the cam-IMU
timeshift is no longer read from calibration.json. It is instead estimated
per-clip from the same left_rectified.mp4 + imu_*.jsonl by cross-correlating
gyroscope magnitude against inter-frame optical-flow magnitude (adapted from
measure_timeshift.py).
"""

import argparse
import json
import pathlib

import cv2
import numpy as np


DATA_ROOT = pathlib.Path("data").resolve()

# MASt3R-Fusion order: [accel_noise_sigma, gyro_noise_sigma,
# accel_bias_rw_sigma, gyro_bias_rw_sigma], matching the Kalibr/imu_utils
# continuous-time noise densities in the raw calibration.json imu section.
IMU_NOISE_KEYS = (
    "accelerometer_noise_density",
    "gyroscope_noise_density",
    "accelerometer_random_walk",
    "gyroscope_random_walk",
)


def _default_output_dir(input_dir: pathlib.Path) -> pathlib.Path:
    try:
        relative_input_dir = input_dir.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Input must be below {DATA_ROOT} when --output is omitted. "
            "Pass --output explicitly for an external capture directory."
        ) from exc
    return DATA_ROOT / "normalized_data" / relative_input_dir


def _find_capture_files(input_dir: pathlib.Path, chunk: str | None):
    """Find the rectified-left video and legacy JSONL IMU file."""
    video_path = input_dir / "left_rectified.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing rectified left video: {video_path}")

    if chunk is not None:
        imu_path = input_dir / f"imu_{chunk}.jsonl"
        if not imu_path.is_file():
            raise FileNotFoundError(f"Missing IMU file: {imu_path}")
        return video_path, imu_path

    imu_files = sorted(input_dir.glob("imu_*.jsonl"))
    if len(imu_files) != 1:
        found = ", ".join(path.name for path in imu_files) or "none"
        raise ValueError(
            f"Expected exactly one imu_*.jsonl in {input_dir}; found {found}. "
            "Pass --chunk when the folder contains multiple chunks."
        )
    return video_path, imu_files[0]


def _imu_noise_from_calibration(calibration: dict) -> list[float] | None:
    """Extract the four continuous-time IMU noise densities from calibration.json.

    Returns them in MASt3R-Fusion order, or None when the calibration has no
    usable imu section.
    """
    imu = calibration.get("imu")
    if not isinstance(imu, dict):
        return None
    try:
        return [float(imu[key]) for key in IMU_NOISE_KEYS]
    except (KeyError, TypeError, ValueError):
        return None


def _find_raw_imu_noise(input_dir: pathlib.Path) -> tuple[list[float] | None, pathlib.Path | None]:
    """Locate the raw rig calibration and read its IMU noise densities.

    The normalized calibration.json drops the imu section, so the raw capture
    calibration (which carries the Kalibr-style noise densities) is preferred.
    The raw path mirrors the normalized path under the same data root; when no
    such mirror exists we fall back to any raw calibration under DATA_ROOT with
    an imu section, since the noise is a hardware property shared by all
    captures of the rig.
    """
    candidates: list[pathlib.Path] = []
    try:
        relative = input_dir.relative_to(DATA_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        swapped = pathlib.Path(*[
            "raw" if part == "normalized" else part for part in relative.parts
        ])
        candidates.append(DATA_ROOT / swapped / "calibration.json")
    candidates.extend(sorted(DATA_ROOT.rglob("calibration.json")))
    seen: set[pathlib.Path] = set()
    for calibration_path in candidates:
        if calibration_path in seen:
            continue
        seen.add(calibration_path)
        try:
            with calibration_path.open("r") as handle:
                calibration = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        noise = _imu_noise_from_calibration(calibration)
        if noise is not None:
            return noise, calibration_path
    return None, None


def _default_imu_noise() -> list[float]:
    """Fall back to the imu_noise baked into config/base_akai.yaml."""
    import yaml

    config_path = pathlib.Path(__file__).resolve().parent / "config/base_akai.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text())
        return [float(value) for value in cfg["ms_opt"]["imu_noise"]]
    except Exception:
        return [0.02, 0.0015, 0.003, 0.00004]


def _camera_imu_transform(calibration: dict) -> np.ndarray:
    """Use the original rig transform when it is present in the JSON."""
    transform = calibration.get("extrinsics", {}).get("T_cam0_imu")
    if transform is None:
        # The supplied normalized schema does not currently include this value.
        print(
            "WARNING: calibration.json has no extrinsics.T_cam0_imu; writing "
            "identity Tic. Supply the rig transform before relying on VIO poses."
        )
        return np.eye(4, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(
            "calibration.extrinsics.T_cam0_imu must be a 4x4 matrix, "
            f"got {transform.shape}"
        )
    return transform


def _estimate_cam_imu_timeshift(
    imu_jsonl: pathlib.Path,
    video_path: pathlib.Path,
    nframes: int = 2400,
    max_lag: float = 0.20,
    windows: int = 4,
    gyr_lsb_dps: float = 16.4,
) -> float:
    """Estimate the cam-IMU offset from THIS clip (adapted from measure_timeshift.py).

    Same physical principle: the gyro and the camera observe the same rotation.
    Correlating gyro-magnitude against inter-frame optical-flow magnitude over a
    range of candidate lags finds the tau that best aligns the two signals.

    Sign convention (matches measure_timeshift.py / akai calibration.json):
        t_imu = t_camera + tau
    i.e. a fsync-derived camera timestamp must be shifted by +tau, not -tau,
    to land on the IMU's own clock. gyr_lsb_dps only rescales gmag by a single
    positive constant across all samples; the z-normalization below cancels
    that out completely, so it has no effect on the recovered tau (verified
    numerically) -- it is set to 16.4 here only so the printed diagnostics use
    physically-correct units, matching the accel/gyro conversion below.
    """
    t_sec, gyro, ft = [], [], []
    with imu_jsonl.open("r") as handle:
        for line in handle:
            record = json.loads(line)
            t_sec.append(record["t_us"] / 1e6)
            gyro.append((record["gx"], record["gy"], record["gz"]))
            if record.get("fsync_flag", 0) == 1:
                ft.append((record["t_us"] - record.get("fsync_delay_us", 0)) / 1e6)
    t_sec = np.asarray(t_sec, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    gmag = np.linalg.norm(gyro / gyr_lsb_dps * np.pi / 180.0, axis=1)
    ft = np.asarray(ft, dtype=np.float64)
    if len(ft) < 30:
        raise ValueError(
            f"Only {len(ft)} fsync timestamps in {imu_jsonl}; "
            "need at least ~30 to estimate a timeshift."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for timeshift estimation: {video_path}")
    prev = None
    cam, ct = [], []
    frame_idx = 0
    while frame_idx < nframes and frame_idx < len(ft):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (480, 270))
        if prev is not None:
            p0 = cv2.goodFeaturesToTrack(prev, 200, 0.01, 10)
            if p0 is not None:
                p1, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None)
                good = status.ravel() == 1
                if good.sum() > 10:
                    flow = np.linalg.norm((p1 - p0)[good].reshape(-1, 2), axis=1)
                    cam.append(np.median(flow))
                    ct.append(ft[frame_idx])
        prev = gray
        frame_idx += 1
    cap.release()

    cam = np.asarray(cam)
    ct = np.asarray(ct)
    if len(cam) < 30:
        raise ValueError(
            f"Only tracked {len(cam)} usable optical-flow frames in {video_path}; "
            "need at least ~30 to estimate a timeshift."
        )
    camz = (cam - cam.mean()) / cam.std()

    lags = np.arange(-max_lag, max_lag + 1e-9, 0.001)

    def corr_at(tau, cz=camz, cts=ct):
        ga = np.interp(cts + tau, t_sec, gmag)
        ga = (ga - ga.mean()) / ga.std()
        return np.corrcoef(cz, ga)[0, 1]

    cc = np.array([corr_at(lag) for lag in lags])
    tau = float(lags[cc.argmax()])
    peak_corr = float(cc.max())

    window_len = len(ct) // windows
    window_taus = []
    for k in range(windows):
        sl = slice(k * window_len, (k + 1) * window_len)
        if sl.stop - sl.start < 30:
            continue
        ccw = np.array([corr_at(lag, camz[sl], ct[sl]) for lag in lags])
        window_taus.append(lags[ccw.argmax()])

    print(f"[timeshift] frames analysed: {len(cam)}")
    print(f"[timeshift] best tau: {tau:+.3f}s (correlation {peak_corr:.3f})")
    if window_taus:
        spread_ms = float(np.std(window_taus)) * 1000
        print(
            f"[timeshift] reproducibility: {np.mean(window_taus)*1000:+.0f} "
            f"+/- {spread_ms:.0f} ms across {len(window_taus)} windows"
        )
        if spread_ms > 10:
            print(
                "[timeshift] WARNING: >10ms spread across windows. This usually "
                "means some windows had motion dominated by translation "
                "(parallax) rather than rotation, which biases this estimator. "
                "Inspect the per-window values before trusting the aggregate tau."
            )
    return tau


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one normalized Akai rectified-left chunk to MASt3R-Fusion input.",
        epilog=(
            "Example: python prepare_akai_sequence_v2.py "
            "--input data/stage-stereo-pipeline-v2-data/normalized/akai/2026-07-20/"
            "akai-ego-001_2026-07-19_16-48-08/000"
        ),
    )
    parser.add_argument(
        "--input", type=pathlib.Path, required=True, metavar="NORMALIZED_CHUNK_DIR",
        help="Directory containing calibration.json, left_rectified.mp4, and imu_*.jsonl.",
    )
    parser.add_argument(
        "--chunk", type=str, default=None,
        help="Optional IMU chunk identifier (for example 000); inferred when omitted.",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=None, metavar="PREPARED_DIR",
        help="Prepared output directory; defaults to the matching data/normalized_data/ path.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--timeshift-nframes", type=int, default=2400,
        help="Frames to analyse when estimating the cam-IMU timeshift (default: 2400, ~80s at 30fps).",
    )
    parser.add_argument(
        "--timeshift-max-lag", type=float, default=0.20,
        help="Search range +/- seconds for the cam-IMU timeshift estimate (default: 0.20).",
    )
    args = parser.parse_args()

    input_dir = args.input.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Normalized Akai chunk directory does not exist: {input_dir}")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    try:
        video_path, imu_jsonl = _find_capture_files(input_dir, args.chunk)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    calibration_path = input_dir / "calibration.json"
    if not calibration_path.is_file():
        parser.error(f"Missing calibration file: {calibration_path}")
    try:
        output_dir = (
            args.output.expanduser().resolve()
            if args.output is not None
            else _default_output_dir(input_dir)
        )
    except ValueError as exc:
        parser.error(str(exc))
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Prepared output: {output_dir}")

    # 1. Use left rectified calibration only.  No raw intrinsics, distortion,
    # or OpenCV undistort/rectify maps are used in this path.
    with calibration_path.open("r") as handle:
        calibration = json.load(handle)

    # 1b. IMU noise densities: the normalized calibration drops the imu section,
    # so read them from the raw rig calibration (see _find_raw_imu_noise).
    imu_noise = _imu_noise_from_calibration(calibration) or _find_raw_imu_noise(input_dir)[0]
    if imu_noise is None:
        imu_noise = _default_imu_noise()
        print(
            "WARNING: no raw calibration with an imu section found; using "
            f"default imu_noise {imu_noise}. Verify it against the device "
            "calibration before relying on VI/global optimization."
        )
    else:
        print(f"[imu_noise] {imu_noise}")
    try:
        rectified_left = calibration["rectified"]["left"]
        fx, fy, cx, cy = (
            float(rectified_left[key]) for key in ("fx", "fy", "cx", "cy")
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"Invalid calibration.rectified.left: {exc}")
    if fx <= 0 or fy <= 0:
        parser.error("Rectified left fx and fy must be positive")
    try:
        tic = _camera_imu_transform(calibration)
    except ValueError as exc:
        parser.error(str(exc))

    # 2. Cam-IMU timeshift: measured from THIS clip, not read from calibration.json.
    print("Estimating cam-IMU timeshift from gyro/optical-flow correlation...")
    try:
        timeshift_cam_imu_s = _estimate_cam_imu_timeshift(
            imu_jsonl,
            video_path,
            nframes=args.timeshift_nframes,
            max_lag=args.timeshift_max_lag,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    # 3. Retain the original JSONL IMU conversion and fsync timing logic.
    imu_records = []
    frame_times = []
    with imu_jsonl.open("r") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                t_us = record["t_us"]
                ax = record["ax"] / 16384.0 * 9.80665
                ay = record["ay"] / 16384.0 * 9.80665
                az = record["az"] / 16384.0 * 9.80665
                gx = record["gx"] / 16.4 * (np.pi / 180.0)
                gy = record["gy"] / 16.4 * (np.pi / 180.0)
                gz = record["gz"] / 16.4 * (np.pi / 180.0)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                parser.error(f"Invalid IMU record at {imu_jsonl}:{line_number}: {exc}")
            imu_records.append([float(t_us) / 1e6, gx, gy, gz, ax, ay, az])
            if record.get("fsync_flag", 0) == 1:
                t_cam_sec = (float(t_us) - float(record.get("fsync_delay_us", 0))) / 1e6
                # t_imu = t_camera + tau (see _estimate_cam_imu_timeshift docstring) ->
                # ADD the offset to land the fsync timestamp on the IMU clock.
                frame_times.append(t_cam_sec + timeshift_cam_imu_s)

    if not imu_records:
        parser.error(f"No IMU samples found in {imu_jsonl}")
    if len(frame_times) <= 3:
        parser.error("Need more than three fsync camera timestamps in the IMU JSONL")
    imu_records = np.asarray(imu_records, dtype=np.float64)

    # 4. Keep the established last-three-frame trimming.
    num_frames = len(frame_times) - 3
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
    frame_times = frame_times[:num_frames]

    # 5. Save original-layout timestamps and custom_rad IMU CSV.
    with (output_dir / "camera_timestamps.txt").open("w") as handle:
        for timestamp in frame_times:
            handle.write(f"{timestamp:.6f}\n")
    np.savetxt(output_dir / "imu.csv", imu_records, delimiter=",", fmt="%.6f")

    # 6. Resolution comes from the rectified video; rectified calibration itself
    # has only K and must not be replaced by raw calibration metadata.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open rectified MP4 video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine frame size for {video_path}")
    with (output_dir / "camera.yaml").open("w") as handle:
        handle.write(f"width: {width}\n")
        handle.write(f"height: {height}\n")
        handle.write(f"calibration: [{fx:.6f}, {fy:.6f}, {cx:.6f}, {cy:.6f}]\n")
        handle.write("rectified: true\n")
        handle.write("Tic:\n")
        for row in tic:
            handle.write(f"  - [{row[0]:.8f},{row[1]:.8f},{row[2]:.8f},{row[3]:.8f}]\n")

    # 7. Extract already-rectified pixels unchanged.
    print("Extracting rectified-left frames...")
    frame_idx = 0
    while cap.isOpened() and frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = frames_dir / f"{frame_times[frame_idx]:.6f}.jpg"
        if not cv2.imwrite(str(out_path), frame):
            cap.release()
            raise RuntimeError(f"Could not write frame: {out_path}")
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{num_frames} frames...")
    cap.release()
    if frame_idx == 0:
        raise RuntimeError(f"Could not read frames from {video_path}")
    if frame_idx != num_frames:
        print(f"WARNING: video ended after {frame_idx}/{num_frames} frames.")

    with (output_dir / "mast3r_akai_v2.yaml").open("w") as handle:
        handle.write('inherit: "config/base_akai.yaml"\n')
        handle.write("dataset:\n")
        handle.write("  center_principle_point: False\n")
        handle.write("  subsample: 1\n")
        handle.write("  img_downsample: 1\n")
        handle.write("ms_opt:\n")
        handle.write(f"  imu_noise: [{imu_noise[0]:.10g}, {imu_noise[1]:.10g}, {imu_noise[2]:.10g}, {imu_noise[3]:.10g}]\n")
        handle.write("global_opt:\n")
        handle.write(f"  imu_noise: [{imu_noise[0]:.10g}, {imu_noise[1]:.10g}, {imu_noise[2]:.10g}, {imu_noise[3]:.10g}]\n")

    print("Done!")
    print(f"Run with: python main_v2.py --prepared-dir {output_dir} --no-viz")


if __name__ == "__main__":
    main()