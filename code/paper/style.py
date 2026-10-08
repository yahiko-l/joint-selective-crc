"""Shared plotting style and helpers for the scripts that render the paper's tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
OUT_DIR = HERE / "out"

FONT_SIZE = 8

mpl.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
})

COLORS = {
    "ours": "#0072B2",
    "hoeffding": "#D55E00",
    "bernstein": "#009E73",
    "wsr": "#CC79A7",
    "ebh": "#F0E442",
    "scrct": "#56B4E9",
    "score": "#E69F00",
    "neutral": "#888888",
}

MARKERS = {
    "ours": "o",
    "hoeffding": "s",
    "bernstein": "D",
    "wsr": "^",
    "ebh": "v",
}

# Figure widths in inches: one column and the full text width.
WIDTH_SINGLE_COL = 3.39
WIDTH_TWO_COL = 7.07


def load_json(rel_path: str) -> dict:
    """Load a result file given its path relative to results/."""
    with (RESULTS_DIR / rel_path).open() as f:
        return json.load(f)


def save_fig(fig, name: str, fmt: str = "pdf") -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{name}.{fmt}"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")
    return out


def write_tex(name: str, tex: str) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{name}.tex"
    out.write_text(tex.rstrip() + "\n")
    print(f"saved {out}")
    return out
