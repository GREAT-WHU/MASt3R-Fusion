# Render Trajectory Video V3 - User Guide

## Overview

`render_trajectory_video_v3.py` creates synchronized visualization videos showing:
- Source camera feed
- Top-down trajectory view
- 3D trajectory view
- Trajectory information panel

**Key Feature**: Properly handles IMU-camera timeshift and automatically finds source files.

---

## Configuration

### Edit Paths (Top of File)

Open `render_trajectory_video_v3.py` and edit these paths to match your system:

```python
# ============================================================
# EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ============================================================
PROJECT_ROOT = Path("/home/ubuntu/abir/SLAM/master_fusion/MASt3R-Fusion")
DATA_ROOT = PROJECT_ROOT / "data"
NORMALIZED_DATA_ROOT = DATA_ROOT / "normalized_data"
SOURCE_VIDEO_ROOT = DATA_ROOT / "stage-stereo-pipeline-v2-data/normalized/akai"
MASTER_OUTPUT_ROOT = PROJECT_ROOT / "analytics/master_output"
# ============================================================
```

**What each path means:**

- `PROJECT_ROOT`: Your MASt3R-Fusion project directory
- `DATA_ROOT`: Where all data is stored
- `NORMALIZED_DATA_ROOT`: Where prepared sequences are (created by prepare_akai_sequence_v2.py)
- `SOURCE_VIDEO_ROOT`: Where original left_rectified.mp4 videos are located
- `MASTER_OUTPUT_ROOT`: Where SLAM outputs are saved (by main_v2.py)

---

## Usage

### Basic Command

```bash
python analytics/render_trajectory_video_v3.py \
  --output-dir analytics/master_output/akai-001/akai-ego-001_2026-07-14_07-31-29/002
```

This will:
1. ✅ Find source video automatically
2. ✅ Find camera timestamps automatically
3. ✅ Load `report.csv` from output-dir
4. ✅ Create `trajectory_overlay.mp4` in output-dir

### Optional Arguments

```bash
--report PATH           # Custom report.csv location (default: output-dir/report.csv)
--output PATH           # Custom output video path (default: output-dir/trajectory_overlay.mp4)
--output-fps FPS        # Output frame rate (default: source video fps, usually 30)
--start SECONDS         # Start time in camera time (default: auto-detect overlap)
--end SECONDS           # End time in camera time (default: auto-detect overlap)
--render-stride N       # Refresh 3D view every N frames (default: 5, higher=faster but choppier 3D)
```

### Example with Options

```bash
python analytics/render_trajectory_video_v3.py \
  --output-dir analytics/master_output/akai-001/akai-ego-001_2026-07-14_07-31-29/002 \
  --output-fps 15 \
  --start 10.0 \
  --end 60.0 \
  --render-stride 10
```

This renders only seconds 10-60 at 15 fps (faster encoding).

---

## Prerequisites

Before running this script:

1. ✅ **Prepared sequence** exists (from `prepare_akai_sequence_v2.py`)
2. ✅ **SLAM trajectory** generated (from `main_v2.py`)
3. ✅ **report.csv** created from result.txt

### Generate report.csv

If you only have `result.txt`, convert it to `report.csv`:

```bash
python analytics/generate_analytics_v2.py \
  --result analytics/master_output/.../002/result.txt \
  --output analytics/master_output/.../002/report.csv
```

---

## Complete Workflow Example

```bash
# 1. Prepare sequence (extract frames, apply timeshift)
python prepare_akai_sequence_v2.py \
  --input data/stage-stereo-pipeline-v2-data/normalized/akai/akai-001/akai-ego-001_2026-07-14_07-31-29/002-v2

# 2. Run SLAM
python main_v2.py \
  --prepared-dir data/normalized_data/.../002 \
  --no-viz \
  --save_h5

# 3. Generate report (convert result.txt to report.csv)
python analytics/generate_analytics_v2.py \
  --result analytics/master_output/akai-001/akai-ego-001_2026-07-14_07-31-29/002/result.txt \
  --output analytics/master_output/akai-001/akai-ego-001_2026-07-14_07-31-29/002/report.csv

# 4. Render visualization video
python analytics/render_trajectory_video_v3.py \
  --output-dir analytics/master_output/akai-001/akai-ego-001_2026-07-14_07-31-29/002
```

---

## Understanding Synchronization

### How It Works

1. **prepare_akai_sequence_v2.py**:
   - Estimates cam-IMU timeshift (e.g., -140ms)
   - Applies timeshift to all frame timestamps
   - Extracts frames with adjusted timestamps
   - Saves `camera_timestamps.txt`

2. **main_v2.py**:
   - Reads frames in sequential order (0, 1, 2, ...)
   - Frame N has timestamp `camera_timestamps[N]`
   - Generates trajectory with matching timestamps

3. **render_trajectory_video_v3.py**:
   - Loads original `left_rectified.mp4`
   - Loads `camera_timestamps.txt` (with timeshift)
   - Maps: Video frame N ↔ timestamp `camera_timestamps[N]` ↔ trajectory pose
   - **Result**: Perfect synchronization! 🎯

### Why Negative Timestamps?

If camera clock runs ahead of IMU, timestamps can be negative after adjustment.
This is **normal and correct** - the script handles it properly.

### Why Video Doesn't Start at Frame 0?

SLAM systems need initialization (several frames to build initial map).
The trajectory typically starts 1-2 seconds into the video.
The script automatically finds and renders only the overlapping portion.

---

## Troubleshooting

### "Could not find prepared data directory"

**Problem**: Script can't find `camera_timestamps.txt`

**Solution**: Make sure `prepare_akai_sequence_v2.py` was run for this sequence.
Check that `NORMALIZED_DATA_ROOT` path is correct.

### "Could not find source video"

**Problem**: Script can't find `left_rectified.mp4`

**Solution**: Check that `SOURCE_VIDEO_ROOT` path is correct.
Verify the video exists at the expected location.

### "Report file not found"

**Problem**: `report.csv` doesn't exist

**Solution**: Generate it from `result.txt` using `generate_analytics_v2.py`

### "Time range does not overlap"

**Problem**: Trajectory and video timestamps don't overlap

**Solution**: Check that the trajectory was generated from the same prepared sequence.
Verify timestamps in `camera_timestamps.txt` and `report.csv`.

### Video Looks Choppy

**Problem**: 3D view updates too frequently (slow rendering)

**Solution**: Increase `--render-stride` (e.g., `--render-stride 10` or `--render-stride 20`)

---

## Output

**File**: `trajectory_overlay.mp4` (or custom path with `--output`)

**Resolution**: 1280x720 (2x2 grid of 640x360 panels)

**Layout**:
```
┌──────────────┬──────────────┐
│ Camera View  │  Top View    │
├──────────────┼──────────────┤
│ Info Panel   │  3D View     │
└──────────────┴──────────────┘
```

**Frame Rate**: Matches source video (usually 30 fps) or custom with `--output-fps`

---

## Notes

- ⚠️ This renders **estimated** trajectory (no ground truth comparison)
- ⚠️ Red arrow shows net displacement, NOT drift error
- ⚠️ Trajectory accuracy depends on SLAM performance
- ✅ Synchronization is maintained throughout the video
- ✅ Only synchronized portions are rendered (automatic overlap detection)

---

## For Different Systems

If your directory structure is different, edit the path variables at the top of the script:

1. Locate your project root
2. Identify where normalized data is stored
3. Identify where source videos are stored
4. Update the path constants accordingly

The script will then work with your custom structure.

