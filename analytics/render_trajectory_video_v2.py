"""Render synchronized source video, estimated top view, and 3D trajectory.

This tool visualizes the trajectory estimate.  It cannot measure true drift
without ground-truth poses; the red vector in the top view is therefore the
estimated net displacement from the start, not a ground-truth drift error.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYTICS_DIR / "run_1"
DEFAULT_VIDEO = ROOT / "data/stage-stereo-pipeline-v2-data/normalized/akai/2026-07-20/akai-ego-001_2026-07-19_16-48-08/000/left_rectified.mp4"
DEFAULT_TIMESTAMPS = ROOT / "data/normalized_data/stage-stereo-pipeline-v2-data/normalized/akai/2026-07-20/akai-ego-001_2026-07-19_16-48-08/000/camera_timestamps.txt"
DEFAULT_REPORT = OUTPUT_DIR / "report.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "trajectory_overlay.mp4"
PANEL_W, PANEL_H = 640, 360


def nearest_index(timestamps: np.ndarray, timestamp: float) -> int:
    right = int(np.searchsorted(timestamps, timestamp, side="left"))
    if right == 0:
        return 0
    if right >= len(timestamps):
        return len(timestamps) - 1
    return right if timestamps[right] - timestamp < timestamp - timestamps[right - 1] else right - 1


def pose_at(report: pd.DataFrame, timestamp: float) -> np.ndarray:
    columns = ["tx", "ty", "tz"]
    return np.array([np.interp(timestamp, report.timestamp, report[column]) for column in columns])


def coordinate_mapper(points: np.ndarray):
    xy_min, xy_max = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    extent = np.maximum(xy_max - xy_min, 0.02)
    margin = 0.15 * max(extent)
    lower, upper = xy_min - margin, xy_max + margin
    width, height = upper - lower

    def to_pixel(point: np.ndarray) -> tuple[int, int]:
        x = int((point[0] - lower[0]) / width * (PANEL_W - 80) + 55)
        y = int((upper[1] - point[1]) / height * (PANEL_H - 80) + 35)
        return x, y

    return to_pixel, lower, upper


def draw_top_view(report: pd.DataFrame, timestamp: float, to_pixel, elapsed: float) -> np.ndarray:
    panel = np.full((PANEL_H, PANEL_W, 3), 248, dtype=np.uint8)
    cv2.putText(panel, "Top view (estimated XY)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(panel, "red = net displacement, not ground-truth drift", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 180), 1, cv2.LINE_AA)

    completed = report.loc[report.timestamp <= timestamp, ["tx", "ty", "tz"]].to_numpy()
    current = pose_at(report, timestamp)
    path = np.vstack((completed, current)) if len(completed) else current[None]
    pixels = np.array([to_pixel(point) for point in path], dtype=np.int32)
    if len(pixels) > 1:
        cv2.polylines(panel, [pixels], False, (210, 90, 30), 2, cv2.LINE_AA)

    start, current_pixel = to_pixel(report.loc[0, ["tx", "ty", "tz"]].to_numpy()), to_pixel(current)
    cv2.circle(panel, start, 6, (40, 170, 40), -1, cv2.LINE_AA)
    cv2.circle(panel, current_pixel, 7, (30, 50, 230), -1, cv2.LINE_AA)
    cv2.arrowedLine(panel, start, current_pixel, (30, 30, 220), 2, cv2.LINE_AA, tipLength=0.04)
    displacement = float(np.linalg.norm(current[:2] - report.loc[0, ["tx", "ty"]].to_numpy()))
    cv2.putText(panel, f"t = {elapsed:6.2f} s", (18, 337), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(panel, f"net displacement: {displacement:.3f} m", (250, 337), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 180), 1, cv2.LINE_AA)
    return panel


def render_3d_panel(report: pd.DataFrame, timestamp: float, limits: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    figure = Figure(figsize=(6.4, 3.6), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111, projection="3d")
    completed = report.loc[report.timestamp <= timestamp, ["tx", "ty", "tz"]].to_numpy()
    current = pose_at(report, timestamp)
    path = np.vstack((completed, current)) if len(completed) else current[None]
    axes.plot(path[:, 0], path[:, 1], path[:, 2], color="#1f77b4", linewidth=2, label="estimated path")
    axes.scatter(*path[0], c="green", s=35, label="start")
    axes.scatter(*current, c="red", s=40, label="current")
    lower, upper = limits
    axes.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]), title="3D trajectory (estimated)")
    axes.set_xlabel("X (m)", labelpad=-4)
    axes.set_ylabel("Y (m)", labelpad=-4)
    axes.set_zlabel("Z (m)", labelpad=-2)
    axes.view_init(elev=24, azim=-58)
    axes.legend(loc="upper left", fontsize=7)
    figure.tight_layout(pad=0.5)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def make_info_panel(timestamp: float, start_time: float, pose: np.ndarray, source_frame: int, source_count: int) -> np.ndarray:
    panel = np.full((PANEL_H, PANEL_W, 3), 30, dtype=np.uint8)
    lines = [
        "MASt3R-Fusion v2 synchronized playback",
        f"source frame: {source_frame + 1:,} / {source_count:,}",
        f"camera timestamp: {timestamp:.6f} s",
        f"trajectory elapsed: {timestamp - start_time:.3f} s",
        f"position (m): [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}]",
        "",
        "Trajectory is an estimate.",
        "No ground truth was supplied, so accuracy drift",
        "cannot be measured or claimed in this video.",
    ]
    y = 46
    for line in lines:
        cv2.putText(panel, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        y += 31
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a synchronized MASt3R-Fusion trajectory video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--timestamps", type=Path, default=DEFAULT_TIMESTAMPS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-fps", type=float, default=None, help="Playback/output frame rate. Defaults to the source video frame rate.")
    parser.add_argument("--start", type=float, default=None, help="Optional camera-time start in seconds.")
    parser.add_argument("--end", type=float, default=None, help="Optional camera-time end in seconds.")
    parser.add_argument("--render-stride", type=int, default=5, help="Refresh 3D panel every N output frames.")
    args = parser.parse_args()

    if args.output_fps is not None and args.output_fps <= 0:
        parser.error("--output-fps must be positive.")
    if args.render_stride <= 0:
        parser.error("--render-stride must be positive.")
    for path in (args.video, args.timestamps, args.report):
        if not path.is_file():
            parser.error(f"Required input not found: {path}")

    report = pd.read_csv(args.report).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    required_columns = {"timestamp", "tx", "ty", "tz"}
    if not required_columns.issubset(report.columns) or report.empty:
        parser.error(f"{args.report} must contain {sorted(required_columns)} and at least one pose.")
    camera_timestamps = np.loadtxt(args.timestamps, dtype=float)
    if camera_timestamps.ndim != 1 or len(camera_timestamps) < 2:
        parser.error("Camera timestamp file must contain at least two timestamps.")

    start_time = max(report.timestamp.iloc[0], camera_timestamps[0]) if args.start is None else args.start
    end_time = min(report.timestamp.iloc[-1], camera_timestamps[-1]) if args.end is None else args.end
    if end_time <= start_time:
        parser.error("The requested time range does not overlap the trajectory and camera timestamps.")
    start_frame, end_frame = nearest_index(camera_timestamps, start_time), nearest_index(camera_timestamps, end_time)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if args.output_fps is None:
        if source_fps > 0:
            args.output_fps = source_fps
            print(f"Output FPS set to source video rate: {args.output_fps:g} fps")
        else:
            args.output_fps = 30.0
            print(f"Warning: could not read source FPS; falling back to {args.output_fps:g} fps")
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame >= source_count:
        raise RuntimeError("Timestamp file has more entries than the source video.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (PANEL_W * 2, PANEL_H * 2))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    points = report[["tx", "ty", "tz"]].to_numpy()
    to_pixel, lower_xy, upper_xy = coordinate_mapper(points)
    xyz_min, xyz_max = points.min(axis=0), points.max(axis=0)
    xyz_margin = max(float((xyz_max - xyz_min).max()) * 0.15, 0.02)
    limits = (xyz_min - xyz_margin, xyz_max + xyz_margin)
    next_output_time, output_index, last_3d = camera_timestamps[start_frame], 0, None
    print(f"Rendering source frames {start_frame}–{end_frame} ({start_time:.3f}–{end_time:.3f} s) ...")

    for frame_index in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode source frame {frame_index}.")
        timestamp = camera_timestamps[frame_index]
        if timestamp + 1e-9 < next_output_time:
            continue
        next_output_time += 1.0 / args.output_fps
        camera_panel = cv2.resize(frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        pose = pose_at(report, timestamp)
        if last_3d is None or output_index % args.render_stride == 0:
            last_3d = render_3d_panel(report, timestamp, limits)
        top_panel = draw_top_view(report, timestamp, to_pixel, timestamp - start_time)
        info_panel = make_info_panel(timestamp, start_time, pose, frame_index, source_count)
        writer.write(np.vstack((np.hstack((camera_panel, top_panel)), np.hstack((info_panel, last_3d)))))
        output_index += 1
        if output_index % 100 == 0:
            print(f"  wrote {output_index} frames ({timestamp - start_time:.1f} / {end_time - start_time:.1f} s)")

    capture.release()
    writer.release()
    print(f"Done: {args.output} ({output_index} frames at {args.output_fps:g} fps)")


if __name__ == "__main__":
    main()
