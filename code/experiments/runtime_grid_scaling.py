"""Certifier runtime versus grid size on the ImageNet surface.

Times one `certify_grid` call on the ImageNet ResNet-50 (IMAGENET1K_V2) surface
for grids of m = 35, 100, 500 and 1000 (lambda, tau) pairs at two certification
sizes, on a single CPU core: two warm-up calls, then the median wall-clock of
five timed calls.

The m = 35 grid is the paper's 7 x 5 grid; the finer grids keep the same five
tau values and place 20, 100 or 200 lambda values geometrically on the same
[0.001, 0.2] range. The timed call covers the certifier only; the time to build
the per-pair loss, acceptance and value arrays from cached logits is recorded
separately.

Run single-threaded, for example
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python experiments/runtime_grid_scaling.py --out runtime_grid_scaling.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selective_crc import certify_grid  # noqa: E402
from experiments import cifar100  # noqa: E402

# Dataset cache root. Point SCORC_DATA_DIR at the directory that holds
# imagenet_data/, imagenet_v2_data/, cifar100_data/, coco_data/, ade20k_data/.
# Defaults to the bundled data/ directory next to this script.
DATA_ROOT = os.environ.get(
    "SCORC_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ablation_supplement"
LOGITS = f"{DATA_ROOT}/imagenet_data/val_logits.npy"
LABELS = f"{DATA_ROOT}/imagenet_data/val_labels.npy"

N_WARM, N_REPEAT = 2, 5
N_CERTS = (5000, 33000)
PAPER_LAMBDA = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
T = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
N_LAMBDA = {35: 7, 100: 20, 500: 100, 1000: 200}
PARAMS = dict(alpha=0.05, pi_min=0.01, delta=0.05, c=0.1, V=1.0, B=1.0)
N_CLASSES = 1000


def lambda_grid(m: int) -> np.ndarray:
    k = N_LAMBDA[m]
    return PAPER_LAMBDA if k == len(PAPER_LAMBDA) else np.geomspace(0.001, 0.2, k)


def build_arrays(logits, labels, Lambda):
    """Per-pair (L, A, v) arrays of the ImageNet surface, columns lambda-major."""
    m_tau = len(T)
    contains_Y, set_size = cifar100._compute_contains_and_size(logits, labels, Lambda, N_CLASSES)
    L = np.repeat((~contains_Y).astype(np.float64), m_tau, axis=1)
    A = np.tile(cifar100.construct_acceptance(logits, T), (1, len(Lambda)))
    v = np.repeat(contains_Y / np.maximum(set_size, 1), m_tau, axis=1)
    return L, A, v


def median_time(fn, *args, **kwargs):
    for _ in range(N_WARM):
        fn(*args, **kwargs)
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), [float(t) for t in times]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main(out_name: str):
    labels_all = np.load(LABELS).astype(np.int64)
    logits_all = np.load(LOGITS).astype(np.float64)
    env = {
        "cpu": cpu_model(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "loadavg_start": os.getloadavg(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    print(json.dumps(env), flush=True)

    rows = []
    for n_cert in N_CERTS:
        idx = np.random.default_rng(42).choice(len(labels_all), size=n_cert, replace=False)
        logits, labels = logits_all[idx], labels_all[idx]
        for m in N_LAMBDA:
            Lambda = lambda_grid(m)
            t_build, _ = median_time(build_arrays, logits, labels, Lambda)
            L, A, v = build_arrays(logits, labels, Lambda)
            assert L.shape == (n_cert, m)
            t_cert, runs = median_time(certify_grid, L, A, v, check_sample_size=False, **PARAMS)
            rows.append({"n_cert": n_cert, "m": m, "n_lambda": len(Lambda),
                         "certify_ms": 1e3 * t_cert, "certify_runs_ms": [1e3 * t for t in runs],
                         "build_arrays_ms": 1e3 * t_build})
            print(f"n_cert={n_cert:6d} m={m:5d}  certify {1e3 * t_cert:9.2f} ms   "
                  f"build arrays {1e3 * t_build:9.1f} ms", flush=True)

    env["loadavg_end"] = os.getloadavg()
    out = {"protocol": {"n_warm": N_WARM, "n_repeat": N_REPEAT, "statistic": "median",
                        "timed_call": "certify_grid(..., check_sample_size=False)",
                        "cert_subset": "default_rng(42).choice(50000, n_cert, replace=False)",
                        "tau": T.tolist(), "lambda_range": [0.001, 0.2], "params": PARAMS},
           "env": env, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / out_name
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runtime_grid_scaling.json")
    main(parser.parse_args().out)
