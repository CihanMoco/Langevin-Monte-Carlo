"""Shared plumbing for the experiment scripts: path bootstrap, output
directories, and small helpers so step_size_sweep.py / friction_sweep.py /
dimension_scaling.py / precision_scaling.py don't repeat themselves.

Not part of the public API described in the README's ``src/`` package -
just infrastructure local to ``experiments/``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # headless: scripts only need to *save* figures
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "plots"
RESULTS_DIR = REPO_ROOT / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(name: str, data: Dict[str, Any]) -> Path:
    """JSON-dump a results dict to results/<name>.json (numpy-safe)."""
    ensure_output_dirs()
    path = RESULTS_DIR / f"{name}.json"

    def default(o):
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serialisable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=default)
    return path


def save_figure(fig, name: str) -> Path:
    ensure_output_dirs()
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(REPO_ROOT)}")
    return path


# A small fixed colour/marker scheme so every figure in the repo is
# visually consistent.
SAMPLER_STYLE = {
    "ULA": dict(color="#1f77b4", marker="o"),
    "MALA": dict(color="#d62728", marker="s"),
    "KLMC": dict(color="#2ca02c", marker="^"),
}
