#!/usr/bin/env python3
"""Run a custom SR simulation and save fit figures.

This script bridges into the external SR codebase, runs one heterogeneous
simulation, and writes:

- a combined survival + hazard PNG
- a waterfall PNG of alive-only X distributions by age bin
- a JSON file with the parameters used

Example:
    python3 src/run_custom_sr_fit.py \
        --label male_custom \
        --eta 1.3 \
        --beta 173.9 \
        --epsilon 0.833 \
        --xc 1.23 \
        --xc-variation 0.27 \
        --kappa 0.5 \
        --dt 0.025
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


DEFAULT_SR_ROOT = Path(
    "/Users/benshenhar/Library/CloudStorage/GoogleDrive-benshenhar@gmail.com/"
    "My Drive/Weizmann/Alon Lab/Aging/python"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="male_sr_fit")
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--xc", type=float, required=True)
    parser.add_argument("--xc-variation", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--tmax", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--save-times", type=float, default=1.0)
    parser.add_argument("--h-ext", type=float, default=0.0)
    parser.add_argument(
        "--out-dir",
        default="output/sr_fits_results",
        help="Output directory for PNG and JSON artifacts.",
    )
    parser.add_argument(
        "--sr-root",
        default=str(DEFAULT_SR_ROOT),
        help="Root of the external SR python project.",
    )
    return parser.parse_args()


def load_sr_modules(sr_root: Path):
    if not sr_root.exists():
        raise FileNotFoundError(f"SR root does not exist: {sr_root}")

    sys.path.insert(0, str(sr_root))
    from ageing_packages.utils import sr_utils as utils  # type: ignore
    from ageing_packages.SR_models.plotting import SR_plotting  # type: ignore

    return utils, SR_plotting


def build_params_dict(utils, args: argparse.Namespace) -> dict:
    base_params = {
        "eta": args.eta,
        "beta": args.beta,
        "kappa": args.kappa,
        "epsilon": args.epsilon,
        "Xc": args.xc,
    }

    return utils.create_param_distribution_dict(
        params="Xc",
        std=args.xc_variation,
        n=args.n,
        dist_type="gaussian",
        params_dict=base_params,
        family="None",
    )


def get_alive_values_in_bin(
    tspan: np.ndarray,
    paths: np.ndarray,
    death_times: np.ndarray,
    age_lo: float,
    age_hi: float,
) -> np.ndarray:
    age_mid = (age_lo + age_hi) / 2.0
    time_idx = int(np.argmin(np.abs(tspan - age_mid)))
    alive_mask = death_times > age_mid

    if not np.any(alive_mask):
        return np.array([])

    return np.asarray(paths[alive_mask, time_idx], dtype=float)


def save_survival_hazard_figure(sim, sr_plotting_cls, args: argparse.Namespace, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plotter = sr_plotting_cls(sim)

    plotter.plot_survival(ax=axes[0], color="#2166ac", linewidth=2.2)
    axes[0].set_title("Survival", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Survival")
    axes[0].set_xlim(args.dt, args.tmax)
    axes[0].set_ylim(0, 1.02)

    plotter.plot_hazard(ax=axes[1], color="#b35806", linewidth=2.2, bandwidth=3)
    axes[1].set_title("Hazard", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel("Hazard (1/year)")
    axes[1].set_xlim(args.dt, args.tmax)

    param_lines = [
        f"eta = {args.eta}",
        f"beta = {args.beta}",
        f"epsilon = {args.epsilon}",
        f"Xc mean = {args.xc}",
        f"Xc CV-like std frac = {args.xc_variation}",
        f"kappa = {args.kappa}",
        f"h_ext = {args.h_ext}",
        f"n = {args.n:,}",
        f"dt = {args.dt}",
    ]
    axes[1].text(
        1.03,
        0.98,
        "\n".join(param_lines),
        transform=axes[1].transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.suptitle(f"SR fit: {args.label}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_waterfall_figure(sim, args: argparse.Namespace, out_path: Path) -> None:
    tspan = np.asarray(sim.tspan, dtype=float)
    paths = np.asarray(sim.paths, dtype=float)
    death_times = np.asarray(sim.death_times, dtype=float)

    bin_edges = list(range(10, 101, 10))
    bin_pairs = list(zip(bin_edges[:-1], bin_edges[1:]))
    bin_labels = [f"{lo}-{hi}" for lo, hi in bin_pairs]
    distributions = [
        get_alive_values_in_bin(tspan, paths, death_times, age_lo=lo, age_hi=hi)
        for lo, hi in bin_pairs
    ]

    colors = cm.viridis(np.linspace(0.15, 0.95, len(bin_labels)))
    fig, axes = plt.subplots(
        len(bin_labels),
        1,
        figsize=(8, 1.6 * len(bin_labels)),
        sharex=True,
        gridspec_kw={"hspace": -0.3},
    )

    x_max = max(
        (np.percentile(values, 99) for values in distributions if len(values) > 0),
        default=1.0,
    )
    x_grid = np.linspace(0, x_max * 1.05, 400)
    xc_mean = float(np.mean(np.asarray(args.xc)))

    for ax, values, label, color in zip(axes, distributions, bin_labels, colors):
        ax.set_yticks([])
        ax.patch.set_alpha(0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if len(values) < 10:
            ax.text(
                0.5,
                0.5,
                f"{label} (n<10)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="grey",
            )
            continue

        kde = gaussian_kde(values, bw_method=0.3)
        density = kde(x_grid)
        density = density / density.max()

        ax.fill_between(
            x_grid,
            0,
            density,
            color=color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.plot(x_grid, density, color="black", linewidth=0.8)
        ax.axvline(xc_mean, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylim(0, 1.3)
        ax.text(
            -0.01,
            0.3,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="center",
        )

    axes[-1].set_xlabel("SR model X value", fontsize=14)
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].tick_params(axis="x", labelsize=11)
    axes[0].text(
        xc_mean,
        1.25,
        r"$\langle X_c \rangle$",
        color="red",
        fontsize=11,
        ha="center",
        va="bottom",
    )

    fig.suptitle(
        f"Distribution of X by age bin - {args.label}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metadata(args: argparse.Namespace, out_path: Path) -> None:
    payload = {
        "label": args.label,
        "eta": args.eta,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "xc": args.xc,
        "xc_variation": args.xc_variation,
        "kappa": args.kappa,
        "n": args.n,
        "tmax": args.tmax,
        "dt": args.dt,
        "save_times": args.save_times,
        "h_ext": args.h_ext,
        "note": "h_ext defaulted to 0.0 because no external hazard was specified.",
    }
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sr_root = Path(args.sr_root)
    utils, sr_plotting_cls = load_sr_modules(sr_root)
    params_dict = build_params_dict(utils, args)

    sim = utils.create_sr_simulation(
        n=args.n,
        params_dict=params_dict,
        h_ext=args.h_ext,
        tmax=args.tmax,
        dt=args.dt,
        save_times=args.save_times,
        parallel=True,
        break_early=True,
    )

    survival_hazard_path = out_dir / f"{args.label}_survival_hazard.png"
    waterfall_path = out_dir / f"{args.label}_waterfall.png"
    metadata_path = out_dir / f"{args.label}_params.json"

    save_survival_hazard_figure(sim, sr_plotting_cls, args, survival_hazard_path)
    save_waterfall_figure(sim, args, waterfall_path)
    save_metadata(args, metadata_path)

    print(f"Saved -> {survival_hazard_path}")
    print(f"Saved -> {waterfall_path}")
    print(f"Saved -> {metadata_path}")


if __name__ == "__main__":
    main()
