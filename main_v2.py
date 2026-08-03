"""Run the original MASt3R-Fusion entry point on rectified Akai v2 data.

This selects files created by prepare_akai_sequence_v2.py and preserves the
rectified-left pixels and intrinsics exactly. All other main.py flags pass
through unchanged.
"""

import argparse
import pathlib
import runpy
import sys
import yaml

import numpy as np

from mast3r_fusion.config import config
from mast3r_fusion.dataloader import Intrinsics


DATA_ROOT = pathlib.Path("data").resolve()


def _default_prepared_dir(input_dir: pathlib.Path) -> pathlib.Path:
    try:
        relative_input_dir = input_dir.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Input must be below {DATA_ROOT} when --prepared-dir is omitted."
        ) from exc
    return DATA_ROOT / "normalized_data" / relative_input_dir


def _install_rectified_calibration_loader() -> None:
    """Load the supplied K directly and skip a second OpenCV remap."""

    def from_rectified_calib(
        img_size, width, height, calibration, always_undistort=False,
        model="pinhole", scale=1, H_new=None,
    ):
        if not config["use_calib"] and not always_undistort:
            return None
        fx, fy, cx, cy = (float(value) for value in calibration[:4])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        # None maps explicitly identify pixels that are already rectified.
        return Intrinsics(img_size, width, height, K, K.copy(), np.zeros(4), None, None)

    def remap_rectified(self, image):
        if self.mapx is None or self.mapy is None:
            return image
        import cv2
        return cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)

    Intrinsics.from_calib = staticmethod(from_rectified_calib)
    Intrinsics.remap = remap_rectified


def _has_option(arguments: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


def _create_output_directory(prepared_dir: pathlib.Path) -> pathlib.Path:
    """Create output directory structure matching the prepared data path.
    
    Example:
        Input: data/normalized_data/.../akai-001/akai-ego-001_2026-07-14_07-31-29/002
        Output: analytics/master_output/akai-001/akai-ego-001_2026-07-14_07-31-29/002
    """
    # Try to extract the relative path structure from normalized_data
    try:
        # Find the normalized_data part in the path
        parts = prepared_dir.parts
        if "normalized_data" in parts:
            idx = parts.index("normalized_data")
            # Get everything after normalized_data (excluding intermediate dirs like stage-stereo-pipeline-v2-data/normalized/akai)
            remaining_parts = parts[idx + 1:]
            
            # Skip generic intermediate directories and keep the meaningful structure
            # Look for pattern: skip until we find a directory that looks like akai-XXX
            meaningful_parts = []
            found_meaningful = False
            for part in remaining_parts:
                if not found_meaningful and part.startswith("akai-"):
                    found_meaningful = True
                if found_meaningful:
                    meaningful_parts.append(part)
            
            if meaningful_parts:
                relative_path = pathlib.Path(*meaningful_parts)
            else:
                # Fallback: use last 3 parts of the path
                relative_path = pathlib.Path(*parts[-3:])
        else:
            # Fallback: use last 3 parts of the path
            relative_path = pathlib.Path(*prepared_dir.parts[-3:])
    except (ValueError, IndexError):
        # Fallback: use the last part of the path (sequence name)
        relative_path = pathlib.Path(prepared_dir.name)
    
    # Create output directory
    output_dir = pathlib.Path("analytics/master_output") / relative_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MASt3R-Fusion on an Akai v2 left_rectified.mp4 sequence."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prepared-dir", type=pathlib.Path, metavar="PREPARED_DIR",
        help="Directory created by prepare_akai_sequence_v2.py.",
    )
    source.add_argument(
        "--input", type=pathlib.Path, metavar="NORMALIZED_CHUNK_DIR",
        help="Uses the matching data/normalized_data/ prepared output directory.",
    )
    wrapper_args, main_args = parser.parse_known_args()

    try:
        prepared_dir = (
            wrapper_args.prepared_dir.expanduser().resolve()
            if wrapper_args.prepared_dir is not None
            else _default_prepared_dir(wrapper_args.input.expanduser().resolve())
        )
    except ValueError as exc:
        parser.error(str(exc))
    
    # Create output directory structure
    output_dir = _create_output_directory(prepared_dir)
    print(f"Output directory: {output_dir}")

    required_files = {
        "frames": prepared_dir / "frames",
        "camera calibration": prepared_dir / "camera.yaml",
        "camera timestamps": prepared_dir / "camera_timestamps.txt",
        "IMU": prepared_dir / "imu.csv",
        "Akai config": prepared_dir / "mast3r_akai_v2.yaml",
    }
    missing = [f"{label}: {path}" for label, path in required_files.items() if not path.exists()]
    if missing:
        parser.error(
            "Prepared sequence is incomplete. Run prepare_akai_sequence_v2.py first.\n"
            + "\n".join(f"  - {item}" for item in missing)
        )

    # Create a temporary config file with absolute inherit paths
    import yaml
    project_root = pathlib.Path(__file__).parent.resolve()
    
    # Recursively resolve all inherit chains to absolute paths
    def resolve_config_with_absolute_inherits(config_path: pathlib.Path) -> dict:
        """Load config and recursively make all inherit paths absolute."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        if "inherit" in config_data:
            inherit_path = config_data["inherit"]
            # Make inherit path absolute
            if not pathlib.Path(inherit_path).is_absolute():
                abs_inherit = project_root / inherit_path
            else:
                abs_inherit = pathlib.Path(inherit_path)
            
            # Recursively resolve parent config
            parent_config = resolve_config_with_absolute_inherits(abs_inherit)
            # Update inherit to absolute path
            config_data["inherit"] = str(abs_inherit)
        
        return config_data
    
    try:
        config_data = resolve_config_with_absolute_inherits(required_files["Akai config"])
    except Exception as e:
        parser.error(f"Error loading config chain: {e}")
    
    # Write temporary config to output directory
    temp_config = output_dir / "temp_config.yaml"
    with open(temp_config, "w") as f:
        yaml.dump(config_data, f)
    
    # Convert all paths to absolute paths
    managed_options = {
        "--dataset": str(required_files["frames"].resolve()),
        "--config": str(temp_config.resolve()),
        "--calib": str(required_files["camera calibration"].resolve()),
        "--imu_path": str(required_files["IMU"].resolve()),
        "--stamp_path": str(required_files["camera timestamps"].resolve()),
    }
    
    # Set output paths to absolute paths in output directory
    if not _has_option(main_args, "--result_path"):
        result_path = output_dir / "result.txt"
        managed_options["--result_path"] = str(result_path.resolve())
        print(f"Result path: {result_path.resolve()}")
    
    # Add save_h5 path handling - need to modify main.py behavior or pass absolute path
    # For now, we'll create a wrapper that sets the working directory context
    
    conflicts = [option for option in managed_options if _has_option(main_args, option)]
    if conflicts:
        parser.error("main_v2.py selects v2 inputs itself; do not pass " + ", ".join(conflicts))

    _install_rectified_calibration_loader()
    main_path = pathlib.Path(__file__).with_name("main.py").resolve()
    
    # Build sys.argv with all absolute paths
    sys.argv = [str(main_path)]
    for option, value in managed_options.items():
        sys.argv.extend((option, value))
    sys.argv.extend(main_args)
    
    # Monkey-patch the graph save and h5 file creation to use output directory
    import os
    import sys as sys_module
    
    # Store output directory in environment for main.py to use
    os.environ['MAST3R_OUTPUT_DIR'] = str(output_dir.resolve())
    
    print(f"Outputs (result.txt, graph.pkl, data.h5) will be saved to: {output_dir}")
    
    # Run in original directory with all absolute paths
    runpy.run_path(str(main_path), run_name="__main__")


if __name__ == "__main__":
    main()
