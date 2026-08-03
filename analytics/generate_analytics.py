import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

# Paths
RESULT_FILE = "/home/ubuntu/abir/SLAM/master_fusion/MASt3R-Fusion/logs/frames.txt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(RESULT_FILE):
        print(f"Error: Could not find {RESULT_FILE}")
        sys.exit(1)

    print(f"Reading data from {RESULT_FILE}...")
    
    # Read the data
    data = []
    with open(RESULT_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                try:
                    # status=0 is the ordered per-frame tracking stream;
                    # status=1 is an asynchronous keyframe update.
                    if len(parts) >= 17 and int(parts[16]) != 0:
                        continue
                    ts = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    data.append([ts, tx, ty, tz, qx, qy, qz, qw])
                except ValueError:
                    continue

    if not data:
        print("No valid data found.")
        sys.exit(1)

    df = pd.DataFrame(data, columns=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

    print(f"Loaded {len(df)} frames. Computing metrics...")

    # Calculate Euler angles (Roll, Pitch, Yaw)
    quats = df[["qx", "qy", "qz", "qw"]].values
    r = R.from_quat(quats)
    euler = r.as_euler('xyz', degrees=True)
    df["roll"] = euler[:, 0]
    df["pitch"] = euler[:, 1]
    df["yaw"] = euler[:, 2]

    # Calculate speeds
    # Velocity = distance / time
    df["dt"] = df["timestamp"].diff()
    df["dx"] = df["tx"].diff()
    df["dy"] = df["ty"].diff()
    df["dz"] = df["tz"].diff()
    df["dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2 + df["dz"]**2)
    
    # Avoid division by zero
    dt_safe = df["dt"].replace(0, np.nan)
    df["speed"] = df["dist"] / dt_safe
    df["speed"] = df["speed"].fillna(0) # First row will be 0

    # 1. Save Report CSV
    csv_path = os.path.join(OUTPUT_DIR, "report.csv")
    report_df = df[["timestamp", "tx", "ty", "tz", "roll", "pitch", "yaw", "speed"]]
    report_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✓ Saved: {csv_path}")

    # 2. Plot Top View (X-Y)
    top_view_path = os.path.join(OUTPUT_DIR, "top_view.png")
    plt.figure(figsize=(10, 8))
    plt.plot(df["tx"], df["ty"], label="Trajectory", color='b')
    plt.scatter(df["tx"].iloc[0], df["ty"].iloc[0], c='g', marker='o', s=100, label='Start')
    plt.scatter(df["tx"].iloc[-1], df["ty"].iloc[-1], c='r', marker='x', s=100, label='End')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Top View Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(top_view_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {top_view_path}")

    # 3. Plot Speed
    speed_path = os.path.join(OUTPUT_DIR, "speed.png")
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"] - df["timestamp"].iloc[0], df["speed"], color='orange', label="Speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(speed_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {speed_path}")

    # Print Speed Statistics
    print("\nSpeed Statistics")
    print("-" * 40)
    print(f"Average Speed : {df['speed'].mean():.3f} m/s")
    print(f"Maximum Speed : {df['speed'].max():.3f} m/s")
    print(f"Minimum Speed : {df['speed'].min():.3f} m/s")
    print(f"Median Speed  : {df['speed'].median():.3f} m/s")

    # 4. Plot 3D View (Interactive HTML)
    html_path = os.path.join(OUTPUT_DIR, "trajectory_3d.html")
    fig = go.Figure(data=[go.Scatter3d(
        x=df['tx'],
        y=df['ty'],
        z=df['tz'],
        mode='lines',
        line=dict(
            color=df['timestamp'],
            colorscale='Viridis',
            width=4
        ),
        name='Trajectory'
    )])

    # Add start and end points
    fig.add_trace(go.Scatter3d(
        x=[df['tx'].iloc[0]], y=[df['ty'].iloc[0]], z=[df['tz'].iloc[0]],
        mode='markers', marker=dict(size=8, color='green'), name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[df['tx'].iloc[-1]], y=[df['ty'].iloc[-1]], z=[df['tz'].iloc[-1]],
        mode='markers', marker=dict(size=8, color='red'), name='End'
    ))

    fig.update_layout(
        title="3D Trajectory",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_html(html_path)
    print(f"✓ Saved: {html_path}")

    print("\nAnalytics generation complete!")

if __name__ == "__main__":
    main()
