"""B4 — CIFAR-100 PC2c sanity / honest-disclosure experiment.

Small CIFAR-100 sanity configuration (n_cert=500, pi_min=0.10, 1 seed).

Ground-truth labels Y_i come from REAL CIFAR-100 test split (downloaded via
torchvision). Model logits come from:
  - 'pretrained': chenyaofo/pytorch-cifar-models ResNet-20 via torch.hub
                  (requires network access)
  - 'synthetic_realistic': synthetic logits that produce ~70% top-1 accuracy
                            against the real Y_i (no network needed; used for
                            sanity when network is unavailable)

The loss L = 1{Y ∉ C_λ(X)} is mis-coverage of the prediction set; the
prediction set C_λ(X) is the smallest set covering >= 1 - λ of the softmax
probability mass.

Usage
-----
    python experiments/cifar100.py --config configs/cifar100_sanity.yaml
"""
# Released here as the shared acceptance-set helper: _compute_contains_and_size and
# construct_acceptance are imported by experiments/imagenet_scale.py. The standalone
# CIFAR-100 sanity entry point below is not part of the released pipeline, and its
# sanity config and per-pair CSV are not bundled.


from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selective_crc import (  # noqa: E402
    certify_grid,
    build_lambda_tau_grid,
    three_split_indices,
    baseline_a_range_hoeffding,
    baseline_b_accepted_bernstein,
)
from selective_crc.certify import INFEASIBLE  # noqa: E402


# --- Data and model loading -------------------------------------------------


def load_cifar100_test_labels(n_subset: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the CIFAR-100 test split labels (and indices) using torchvision.

    Returns (test_indices, test_labels). For the sanity, only labels are needed
    (logits are synthetic). For the pretrained-model path,
    images would also be loaded.
    """
    import torch
    from torchvision.datasets import CIFAR100
    from torchvision import transforms

    # Standard CIFAR-100 normalization (not used for label-only mode)
    transform = transforms.Compose([transforms.ToTensor()])
    cache_dir = os.environ.get("TORCH_DATA_ROOT", str(Path.home() / ".cache" / "torchvision"))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"[cifar100] loading CIFAR-100 test split (cache dir: {cache_dir})...")
    ds = CIFAR100(
        root=cache_dir, train=False, download=True, transform=transform
    )
    labels = np.array(ds.targets, dtype=np.int64)  # shape (10000,)
    print(f"[cifar100] CIFAR-100 test split loaded: {len(labels)} labels")
    return np.arange(len(labels)), labels


def generate_synthetic_realistic_logits(
    Y: np.ndarray, n_classes: int, accuracy_target: float, seed: int
) -> np.ndarray:
    """Draw synthetic logits achieving ~accuracy_target top-1 accuracy against Y,
    with realistic max-softmax distribution (peak typically 0.4-0.95).

    Strategy:
    - Base noise ~ N(0, 1) on all 100 logits.
    - For each sample: with prob accuracy_target, bump true-class logit by
      ~Uniform(5, 9) → max softmax ≈ exp(7)/Σ ≈ 0.85 typical.
      Else, bump a random WRONG class by ~Uniform(4, 8) → max softmax ≈ 0.6-0.85.
    - Also bump 1-3 "runner-up" classes by ~Uniform(1, 3) to create a realistic
      flat tail (matches what a real CIFAR-100 classifier produces).

    Returns
    -------
    logits : np.ndarray, shape (n, n_classes)
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    logits = rng.normal(0, 1.0, size=(n, n_classes))  # base noise

    correct_mask = rng.uniform(0, 1, size=n) < accuracy_target

    for i in range(n):
        if correct_mask[i]:
            top_class = Y[i]
            bump = rng.uniform(5.0, 9.0)
        else:
            top_class = rng.integers(0, n_classes)
            while top_class == Y[i]:
                top_class = rng.integers(0, n_classes)
            bump = rng.uniform(4.0, 8.0)
        logits[i, top_class] += bump

        # Add 1-3 runner-up bumps for a realistic flat tail
        n_runner = rng.integers(1, 4)  # 1, 2, or 3 runner-ups
        runner_classes = rng.choice(
            [c for c in range(n_classes) if c != top_class],
            size=n_runner,
            replace=False,
        )
        for c in runner_classes:
            logits[i, c] += rng.uniform(1.0, 3.0)

    achieved_acc = float((logits.argmax(axis=1) == Y).mean())
    # Diagnose max-softmax distribution for sanity
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    max_softmax = probs.max(axis=1)
    print(f"[cifar100] synthetic logits generated: shape={logits.shape}, "
          f"achieved top-1 acc = {achieved_acc:.3f} (target {accuracy_target})")
    print(f"[cifar100] max-softmax distribution: "
          f"min={max_softmax.min():.3f}, "
          f"25%={np.percentile(max_softmax, 25):.3f}, "
          f"median={np.median(max_softmax):.3f}, "
          f"75%={np.percentile(max_softmax, 75):.3f}, "
          f"max={max_softmax.max():.3f}")
    return logits


def load_pretrained_logits(images, n_classes=100) -> np.ndarray:
    """Stub for pretrained-model logits. Requires network for first call."""
    raise NotImplementedError(
        "pretrained logits via torch.hub chenyaofo/pytorch-cifar-models not yet "
        "wired up; use logits_source: synthetic_realistic for the sanity."
    )


# --- Prediction-set construction (LAC: Least Ambiguous set-valued Classifiers)


def construct_prediction_sets(logits: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
    """For each (i, k), determine which labels are in C_{λ_k}(X_i).

    NOTE — NOT CALLED BY ANY EXPERIMENT: experiments use the memory-efficient
    `_compute_contains_and_size` below (which returns only `contains_Y` and
    `set_size` rather than the full (n, n_classes, m_lambda) boolean tensor).
    Retained as a reference implementation of the explicit set construction.

    The prediction set C_λ(X) is the smallest set s.t. softmax probability
    sums >= 1 - λ (using the standard "smallest cumulative softmax" set,
    a.k.a. LAC / APS-without-randomization).

    Returns
    -------
    contains : np.ndarray of bool, shape (n, n_classes, m_lambda)
        contains[i, c, k] = 1 iff class c is in the prediction set
        C_{Lambda[k]}(X_i).

    Notes
    -----
    For memory efficiency at larger scales, the (n, n_classes, m_lambda) tensor
    could be replaced by storing only the set-size + (i, k) → contains_Y_i bool.
    For sanity (n=500, n_classes=100, m_lambda=7) the tensor is ~1.4M floats:
    fine in RAM.
    """
    n, n_classes = logits.shape
    m_lambda = len(Lambda)

    # Softmax probabilities
    log_probs = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)

    # Sort descending per sample
    sorted_idx = np.argsort(-probs, axis=1)  # (n, n_classes)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)  # (n, n_classes)

    contains = np.zeros((n, n_classes, m_lambda), dtype=bool)
    for k_lambda, lam in enumerate(Lambda):
        threshold = 1.0 - lam
        # Find smallest position where cumsum >= threshold (per row)
        # For each i: position = argmax(cumsum[i] >= threshold), then include all classes
        # at positions <= that index in the sorted order.
        include_count = (cumsum < threshold).sum(axis=1) + 1  # shape (n,)
        include_count = np.clip(include_count, 1, n_classes)
        for i in range(n):
            # Mark first include_count[i] classes as in C
            included = sorted_idx[i, : include_count[i]]
            contains[i, included, k_lambda] = True

    return contains


def _compute_contains_and_size(
    logits: np.ndarray, Y: np.ndarray, Lambda: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute contains_Y (shape (n, m_lambda)) and set_size (shape (n, m_lambda)).

    contains_Y[i, k] = 1 iff Y[i] is in C_{Lambda[k]}(X_i)
    set_size[i, k]   = |C_{Lambda[k]}(X_i)|
    """
    n = len(Y)
    m_lambda = len(Lambda)
    log_probs = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)
    sorted_idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    contains_Y = np.zeros((n, m_lambda), dtype=bool)
    set_size = np.zeros((n, m_lambda), dtype=np.int64)
    rank_of_Y = np.argsort(sorted_idx, axis=1)[np.arange(n), Y]  # position of Y in sorted order

    for k_lambda, lam in enumerate(Lambda):
        threshold = 1.0 - lam
        include_count = (cumsum < threshold).sum(axis=1) + 1
        include_count = np.clip(include_count, 1, n_classes)
        set_size[:, k_lambda] = include_count
        contains_Y[:, k_lambda] = rank_of_Y < include_count
    return contains_Y, set_size


def construct_acceptance(logits: np.ndarray, T: np.ndarray) -> np.ndarray:
    """A[i, k_tau] = 1{max softmax > T[k_tau]}.

    Returns
    -------
    A : np.ndarray of float, shape (n, m_tau)
    """
    log_probs = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)
    g = probs.max(axis=1)  # shape (n,)
    A = (g[:, None] > T[None, :]).astype(np.float64)
    return A


# --- Main run ---------------------------------------------------------------


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    Lambda = np.array(cfg["grid"]["Lambda"], dtype=np.float64)
    T = np.array(cfg["grid"]["T"], dtype=np.float64)
    m_lambda, m_tau = len(Lambda), len(T)
    m = m_lambda * m_tau

    # Load real CIFAR-100 labels (full test split = 10k); we restrict via the 3-split.
    _, Y_full = load_cifar100_test_labels(
        n_subset=10000,  # CIFAR-100 test split is fixed at 10000
        seed=cfg["seeds"]["split_seed"],
    )
    requested_total = cfg["data"]["n_cert"] + cfg["data"]["n_tune"] + cfg["data"]["n_test"]
    if requested_total > len(Y_full):
        raise ValueError(
            f"Requested n_cert + n_tune + n_test = {requested_total} > CIFAR-100 test size {len(Y_full)}."
        )
    # Use the first requested_total indices in the (seed-shuffled) ordering to avoid waste.
    # Actually: we restrict to a random subset of size requested_total using the master seed.
    master_rng = np.random.default_rng(cfg["seeds"]["master"])
    subset_idx = master_rng.choice(len(Y_full), size=requested_total, replace=False)
    Y_all = Y_full[subset_idx]
    n_total = len(Y_all)
    n_classes = 100
    print(f"[cifar100] restricted to {n_total} samples from CIFAR-100 test split (full size {len(Y_full)})")

    # Logits
    if cfg["data"]["logits_source"] == "synthetic_realistic":
        logits_all = generate_synthetic_realistic_logits(
            Y_all,
            n_classes,
            accuracy_target=cfg["data"]["synthetic_accuracy_target"],
            seed=cfg["data"]["logits_seed"],
        )
    elif cfg["data"]["logits_source"] == "pretrained":
        logits_all = load_pretrained_logits(None, n_classes)
    else:
        raise ValueError(f"Unknown logits_source: {cfg['data']['logits_source']}")

    # 3-split
    tune_idx, cert_idx, test_idx = three_split_indices(
        n_total,
        n_tune=cfg["data"]["n_tune"],
        n_cert=cfg["data"]["n_cert"],
        seed=cfg["seeds"]["split_seed"],
    )
    print(f"[cifar100] split: tune={len(tune_idx)}, cert={len(cert_idx)}, test={len(test_idx)}")

    # Build prediction sets and acceptance for cert split
    Y_cert = Y_all[cert_idx]
    logits_cert = logits_all[cert_idx]

    print(f"[cifar100] constructing prediction sets on cert split (n_cert={len(cert_idx)})...")
    contains_Y, set_size = _compute_contains_and_size(
        logits_cert, Y_cert, Lambda, n_classes
    )
    print(f"[cifar100] contains_Y shape: {contains_Y.shape}, mean coverage per λ: {contains_Y.mean(axis=0)}")
    print(f"[cifar100] mean set size per λ: {set_size.mean(axis=0)}")

    # Loss L = 1 - contains_Y (mis-coverage)
    L_lambda = (~contains_Y).astype(np.float64)  # shape (n_cert, m_lambda)
    # Broadcast to full grid: k = k_lambda * m_tau + k_tau
    L_grid = np.repeat(L_lambda, m_tau, axis=1)  # shape (n_cert, m_lambda*m_tau)

    # Acceptance A
    A_lambda_tau = construct_acceptance(logits_cert, T)  # (n_cert, m_tau)
    A_grid = np.tile(A_lambda_tau, (1, m_lambda))  # shape (n_cert, m_lambda*m_tau)

    # Deployment value v: per EXPERIMENT_PLAN §B2 main paper convention:
    #   v(C_λ(X), Y) = 1{Y ∈ C_λ(X)} / |C_λ(X)|  (accuracy / set-size)
    # Bounded in [0, 1] since set_size >= 1 always; matches V=1 in config.
    set_size_safe = np.maximum(set_size, 1)  # avoid div-by-zero (set_size already >= 1)
    v_lambda = contains_Y.astype(np.float64) / set_size_safe.astype(np.float64)
    v_grid = np.repeat(v_lambda, m_tau, axis=1)

    # Build flat (λ, τ) labels for output
    Lambda_flat, T_flat = build_lambda_tau_grid(Lambda, T)

    print(f"[cifar100] running Algorithm 1: n_cert={len(cert_idx)}, m={m}")

    result = certify_grid(
        L_grid, A_grid, v_grid,
        alpha=cfg["parameters"]["alpha"],
        pi_min=cfg["parameters"]["pi_min"],
        delta=cfg["parameters"]["delta"],
        c=cfg["parameters"]["c"],
        V=cfg["parameters"]["V"],
        B=cfg["parameters"]["B"],
        check_sample_size=cfg.get("check_sample_size", True),
    )

    # Baselines
    baseline_a_width = baseline_a_range_hoeffding(
        L_grid, A_grid,
        alpha=cfg["parameters"]["alpha"],
        pi_min=cfg["parameters"]["pi_min"],
        delta=cfg["parameters"]["delta"],
        B=cfg["parameters"]["B"],
    )
    baseline_b_width = baseline_b_accepted_bernstein(
        L_grid, A_grid,
        alpha=cfg["parameters"]["alpha"],
        delta=cfg["parameters"]["delta"],
        B=cfg["parameters"]["B"],
    )

    # Use the held-out test split for "ground truth" R_sel and p_acc
    Y_test = Y_all[test_idx]
    logits_test = logits_all[test_idx]

    # Recompute contains_Y, set_size, A on TEST split for held-out truth proxy
    contains_Y_test, set_size_test = _compute_contains_and_size(
        logits_test, Y_test, Lambda, n_classes
    )
    L_test_lambda = (~contains_Y_test).astype(np.float64)
    L_test_grid = np.repeat(L_test_lambda, m_tau, axis=1)
    A_test_lambda_tau = construct_acceptance(logits_test, T)
    A_test_grid = np.tile(A_test_lambda_tau, (1, m_lambda))
    set_size_test_safe = np.maximum(set_size_test, 1)
    v_test_lambda = contains_Y_test.astype(np.float64) / set_size_test_safe.astype(np.float64)
    v_test_grid = np.repeat(v_test_lambda, m_tau, axis=1)

    # Test-split estimates of R_sel and p_acc per pair (proxy for population truth)
    p_acc_test = A_test_grid.mean(axis=0)
    sum_AL_test = (A_test_grid * L_test_grid).sum(axis=0)
    sum_A_test = A_test_grid.sum(axis=0)
    R_sel_test = np.divide(
        sum_AL_test, sum_A_test, out=np.full_like(sum_AL_test, np.inf),
        where=sum_A_test > 0,
    )
    U_dep_test = (A_test_grid * v_test_grid - cfg["parameters"]["c"] * (1 - A_test_grid)).mean(axis=0)

    # Selected pair report
    if result.selected != INFEASIBLE:
        k_hat = result.selected
        print(f"\n[cifar100] SELECTED: k={k_hat}, λ={Lambda_flat[k_hat]:.4f}, τ={T_flat[k_hat]:.2f}")
        print(f"  CERT-split: EB={result.eb_per_pair[k_hat]:.4f}, p_LCB={result.p_lcb_per_pair[k_hat]:.4f}, U_LCB={result.u_lcb_per_pair[k_hat]:.4f}")
        print(f"  TEST-split (held-out truth proxy):")
        print(f"    R_sel_test = {R_sel_test[k_hat]:.4f} (target ≤ α={cfg['parameters']['alpha']})")
        print(f"    p_acc_test = {p_acc_test[k_hat]:.4f} (target ≥ π_min={cfg['parameters']['pi_min']})")
        print(f"    U_dep_test = {U_dep_test[k_hat]:.4f}")
        vio_r = bool(R_sel_test[k_hat] > cfg["parameters"]["alpha"])
        vio_p = bool(p_acc_test[k_hat] < cfg["parameters"]["pi_min"])
    else:
        print("\n[cifar100] INFEASIBLE — no pair satisfies (EB ≤ 0 ∧ p_LCB ≥ π_min)")
        k_hat = None
        vio_r = vio_p = None

    print(f"\n[cifar100] |Ĝ| = {result.n_certified} / {m}")

    # Per-pair full table
    per_pair = []
    for k in range(m):
        per_pair.append({
            "k": int(k),
            "lambda": float(Lambda_flat[k]),
            "tau": float(T_flat[k]),
            "Z_bar": float(result.z_bar[k]),
            "sigma_Z_sq": float(result.sigma_z_sq[k]),
            "s_count": int(result.s_count[k]),
            "EB": float(result.eb_per_pair[k]),
            "p_LCB": float(result.p_lcb_per_pair[k]),
            "U_LCB": float(result.u_lcb_per_pair[k]),
            "U_bar": float(result.u_bar[k]),
            "in_G_hat": bool(result.in_g_hat_mask[k]),
            "baseline_a_width": float(baseline_a_width[k]),
            "baseline_b_width": float(baseline_b_width[k]) if not np.isnan(baseline_b_width[k]) else None,
            "R_sel_test": float(R_sel_test[k]) if R_sel_test[k] != np.inf else None,
            "p_acc_test": float(p_acc_test[k]),
            "U_dep_test": float(U_dep_test[k]),
        })

    summary = {
        "experiment": cfg["experiment"]["name"],
        "block": cfg["experiment"]["block"],
        "config_path": str(config_path),
        "m": m,
        "m_lambda": m_lambda,
        "m_tau": m_tau,
        "n_cert": len(cert_idx),
        "n_test": len(test_idx),
        "parameters": cfg["parameters"],
        "config_used": result.config,
        "selected_k": k_hat,
        "selected_lambda": float(Lambda_flat[k_hat]) if k_hat is not None else None,
        "selected_tau": float(T_flat[k_hat]) if k_hat is not None else None,
        "is_infeasible": result.is_infeasible,
        "n_certified": result.n_certified,
        "vio_r_on_selected": vio_r,
        "vio_p_on_selected": vio_p,
        "per_pair": per_pair,
    }

    json_path = out_dir / "results.json"
    csv_path = out_dir / "per_pair.csv"
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k", "lambda", "tau", "Z_bar", "sigma_Z_sq", "s_count",
            "EB", "p_LCB", "U_LCB", "U_bar", "in_G_hat",
            "baseline_a_width", "baseline_b_width",
            "R_sel_test", "p_acc_test", "U_dep_test",
        ])
        for row in per_pair:
            writer.writerow([
                row["k"], row["lambda"], row["tau"],
                row["Z_bar"], row["sigma_Z_sq"], row["s_count"],
                row["EB"], row["p_LCB"], row["U_LCB"], row["U_bar"], row["in_G_hat"],
                row["baseline_a_width"], row["baseline_b_width"],
                row["R_sel_test"], row["p_acc_test"], row["U_dep_test"],
            ])

    print(f"\n[cifar100] wrote {json_path}")
    print(f"[cifar100] wrote {csv_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.config)
