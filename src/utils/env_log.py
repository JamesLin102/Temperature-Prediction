"""Environment logging for reproducibility (spec section 9, step 13)."""
from __future__ import annotations

import importlib
import platform
import sys
from datetime import datetime

from . import paths

_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "lightgbm",
    "statsmodels",
    "pmdarima",
    "sdv",
    "ctgan",
    "matplotlib",
    "yaml",
]


def _pkg_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - missing/broken package is informational only
        return "not installed"


def write_environment_log(output_path=None, extra: dict | None = None) -> str:
    """Write Python / package / run info to ``outputs/logs/environment.txt``."""
    if output_path is None:
        output_path = paths.OUTPUTS_LOGS_DIR / "environment.txt"
    output_path = paths.resolve(output_path)
    paths.ensure_dir(output_path.parent)

    lines = [
        f"Generated:  {datetime.now().isoformat(timespec='seconds')}",
        f"Platform:   {platform.platform()}",
        f"Python:     {sys.version.splitlines()[0]}",
        f"Executable: {sys.executable}",
        "",
        "Package versions:",
    ]
    for pkg in _PACKAGES:
        lines.append(f"  {pkg:<12} {_pkg_version(pkg)}")
    if extra:
        lines.append("")
        lines.append("Run info:")
        for key, value in extra.items():
            lines.append(f"  {key}: {value}")

    text = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(output_path)
