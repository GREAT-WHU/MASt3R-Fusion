"""Generate trajectory analytics for MASt3R-Fusion v2 exports.

The v2 trajectory exporter writes one pose per line as either:
    [timestamp] tx ty tz qx qy qz qw
or:
    timestamp tx ty tz qx qy qz qw
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as Rotation


def parse_trajectory(path: Path) -> pd.DataFrame:
    """Load valid v2 trajectory rows, accepting bracketed timestamps."""
    rows = []
    rejected = 0
    with path.open() as trajectory_file:
        for line_number, line in enumerate(trajectory_file, start=1):
            fields = line.split()
            if len(fields) < 8:
                rejected += 1
                continue
            try:
                timestamp = float(fields[0].strip("[]"))
                pose = [float(value) for value in fields[1:8]]
            except ValueError:
                rejected += 1
                continue
            if not np.isfinite([timestamp, *pose]).all():
                rejected += 1
                continue
            rows.append([timestamp, *pose])

    if not rows:
        raise ValueError(f"No valid pose rows found in {path}.")

    data = pd.DataFrame(
        rows, columns=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    ).sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    if rejected:
        print(f"Skipped {rejected} malformed row(s).")
    return data.reset_index(drop=True)


def add_metrics(data: pd.DataFrame) -> pd.DataFrame:
    quaternions = data[["qx", "qy", "qz", "qw"]].to_numpy().copy()
    norms = np.linalg.norm(quaternions, axis=1)
    valid = norms > np.finfo(float).eps
    if not valid.all():
        data = data.loc[valid].copy()
        quaternions = quaternions[valid]
        norms = norms[valid]
        print(f"Dropped {(~valid).sum()} pose(s) with a zero-length quaternion.")
    quaternions /= norms[:, None]

    euler = Rotation.from_quat(quaternions).as_euler("xyz", degrees=True)
    data[["roll", "pitch", "yaw"]] = euler
    data["dt"] = data["timestamp"].diff()
    displacement = data[["tx", "ty", "tz"]].diff()
    data["distance"] = np.linalg.norm(displacement, axis=1)
    valid_dt = data["dt"] > 0
    data["speed"] = np.where(valid_dt, data["distance"] / data["dt"], 0.0)
    data["speed"] = data["speed"].fillna(0.0)
    return data


def save_plots(data: pd.DataFrame, output_dir: Path) -> None:
    top_view_path = output_dir / "top_view.png"
    plt.figure(figsize=(10, 8))
    plt.plot(data["tx"], data["ty"], color="tab:blue", label="Trajectory")
    plt.scatter(data["tx"].iloc[0], data["ty"].iloc[0], c="green", s=80, label="Start")
    plt.scatter(data["tx"].iloc[-1], data["ty"].iloc[-1], c="red", marker="x", s=80, label="End")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Top View Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(top_view_path, dpi=300, bbox_inches="tight")
    plt.close()

    speed_path = output_dir / "speed.png"
    plt.figure(figsize=(12, 6))
    elapsed = data["timestamp"] - data["timestamp"].iloc[0]
    plt.plot(elapsed, data["speed"], color="tab:orange", label="Speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(speed_path, dpi=300, bbox_inches="tight")
    plt.close()

    trajectory_3d_path = output_dir / "trajectory_3d.html"
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(
        x=data["tx"], y=data["ty"], z=data["tz"], mode="lines",
        line={"color": elapsed, "colorscale": "Viridis", "width": 4}, name="Trajectory",
    ))
    for label, color, index in (("Start", "green", 0), ("End", "red", -1)):
        figure.add_trace(go.Scatter3d(
            x=[data["tx"].iloc[index]], y=[data["ty"].iloc[index]], z=[data["tz"].iloc[index]],
            mode="markers", marker={"size": 8, "color": color}, name=label,
        ))
    figure.update_layout(
        title="3D Trajectory",
        scene={"xaxis_title": "X (m)", "yaxis_title": "Y (m)", "zaxis_title": "Z (m)", "aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    figure.write_html(trajectory_3d_path)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate analytics for a MASt3R-Fusion v2 trajectory.")
    parser.add_argument("--trajectory", type=Path, default=repository_root / "logs" / "frames.txt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output_v2")
    args = parser.parse_args()

    if not args.trajectory.is_file():
        parser.error(f"Trajectory file not found: {args.trajectory}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading v2 trajectory from {args.trajectory}...")
    data = add_metrics(parse_trajectory(args.trajectory))
    if data.empty:
        raise ValueError("No poses with valid quaternions remain.")

    report_path = args.output_dir / "report.csv"
    data[["timestamp", "tx", "ty", "tz", "roll", "pitch", "yaw", "speed"]].to_csv(
        report_path, index=False, float_format="%.6f"
    )
    save_plots(data, args.output_dir)

    print(f"Loaded {len(data)} poses.")
    print(f"Duration: {data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]:.3f} s")
    print(f"Average speed: {data['speed'].mean():.3f} m/s")
    print(f"Maximum speed: {data['speed'].max():.3f} m/s")
    print(f"Saved report and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
