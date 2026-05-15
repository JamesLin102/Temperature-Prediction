"""Copy required legacy assets into revised_framework_v2.

This script is the only approved place to read from sibling source folders:
Original Programe/ and New Dataset/. Runtime experiment code must read only
from revised_framework_v2/.
"""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent

SOURCES = {
    WORKSPACE_DIR / "Original Programe" / "data" / "seattle-weather.csv":
        PROJECT_DIR / "data" / "seattle" / "seattle-weather.csv",
    WORKSPACE_DIR / "New Dataset" / "Bias_correction_ucl.csv":
        PROJECT_DIR / "data" / "seoul" / "Bias_correction_ucl.csv",
    WORKSPACE_DIR / "Original Programe" / "plot.py":
        PROJECT_DIR / "reference_legacy" / "plot.py",
    WORKSPACE_DIR / "Original Programe" / "predict.py":
        PROJECT_DIR / "reference_legacy" / "predict.py",
}

NPY_DIRS = {
    WORKSPACE_DIR / "Original Programe" / "data" / "temp_max":
        PROJECT_DIR / "data" / "seattle" / "temp_max",
    WORKSPACE_DIR / "Original Programe" / "data" / "temp_min":
        PROJECT_DIR / "data" / "seattle" / "temp_min",
    WORKSPACE_DIR / "Original Programe" / "data" / "data_plot" / "temp_max":
        PROJECT_DIR / "data" / "legacy_plot_arrays" / "temp_max",
    WORKSPACE_DIR / "Original Programe" / "data" / "data_plot" / "temp_min":
        PROJECT_DIR / "data" / "legacy_plot_arrays" / "temp_min",
}

OUTPUT_DIRS = [
    PROJECT_DIR / "outputs" / "tables",
    PROJECT_DIR / "outputs" / "predictions",
    PROJECT_DIR / "outputs" / "figures",
    PROJECT_DIR / "outputs" / "arrays",
    PROJECT_DIR / "outputs" / "logs",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_npy_dir(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing required source directory: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob("*.npy")):
        shutil.copy2(src, dst_dir / src.name)
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No .npy files found in: {src_dir}")
    return count


def main() -> None:
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)

    for src, dst in SOURCES.items():
        copy_file(src, dst)
        print(f"copied file: {src} -> {dst}")

    for src_dir, dst_dir in NPY_DIRS.items():
        count = copy_npy_dir(src_dir, dst_dir)
        print(f"copied {count} npy files: {src_dir} -> {dst_dir}")

    print("Asset copy complete. Runtime code should use revised_framework_v2 paths only.")


if __name__ == "__main__":
    main()
