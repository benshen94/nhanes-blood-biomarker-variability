#!/usr/bin/env python3
"""Functional-PCA style analysis for NHANES disease age-timescale KM curves."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatError
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


MORTALITY_FILES = [
    "NHANES_1999_2000_MORT_2019_PUBLIC.dat",
    "NHANES_2001_2002_MORT_2019_PUBLIC.dat",
    "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
    "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
    "NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
    "NHANES_2017_2018_MORT_2019_PUBLIC.dat",
]
MORT_BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
NON_DISEASE_COLUMNS = {
    "seqn",
    "cycle_start_year",
    "age_years",
    "sex",
    "pregnant",
    "healthy_flag",
    "exclusion_reason",
}
DISEASE_META = {
    "diabetes": ("Diabetes", "DIQ010"),
    "asthma": ("Asthma", "MCQ010"),
    "kidney": ("Kidney disease", "KIQ022"),
    "liver": ("Liver disease", "MCQ160L/MCQ500/MCQ510A-F"),
    "cancer": ("Cancer/malignancy", "MCQ220"),
    "cvd": ("Any major CVD", "MCQ160B-F composite"),
    "hypertension": ("Hypertension", "BPQ020"),
    "osteoporosis": ("Osteoporosis", "OSQ060"),
    "cataract_operation": ("Cataract operation", "VIQ070"),
    "arthritis": ("Arthritis", "MCQ160A"),
    "heart_failure": ("Congestive heart failure", "MCQ160B"),
    "coronary_heart_disease": ("Coronary heart disease", "MCQ160C"),
    "angina": ("Angina", "MCQ160D"),
    "heart_attack": ("Heart attack", "MCQ160E"),
    "stroke": ("Stroke", "MCQ160F"),
    "emphysema": ("Emphysema", "MCQ160G"),
    "overweight": ("Overweight", "MCQ160J"),
    "chronic_bronchitis": ("Chronic bronchitis", "MCQ160K"),
    "liver_condition": ("Liver condition", "MCQ160L"),
    "thyroid_problem": ("Thyroid problem", "MCQ160M"),
    "still_chronic_bronchitis": ("Still have chronic bronchitis", "MCQ170K"),
    "still_liver_condition": ("Still have liver condition", "MCQ170L"),
    "still_thyroid_problem": ("Still have thyroid problem", "MCQ170M"),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def download_mortality_files(out_dir: Path) -> None:
    ensure_dir(out_dir)
    for fn in MORTALITY_FILES:
        p = out_dir / fn
        if p.exists() and p.stat().st_size > 0:
            continue
        r = requests.get(MORT_BASE_URL + fn, timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)


def load_mortality(mort_dir: Path) -> pd.DataFrame:
    rows = []
    colspecs = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
    names = ["seqn", "eligstat", "mortstat", "ucod_leading", "diabetes_mcod", "hyperten_mcod", "permth_int", "permth_exm"]
    for fn in MORTALITY_FILES:
        p = mort_dir / fn
        m = pd.read_fwf(p, colspecs=colspecs, names=names, dtype=str)
        m["seqn"] = to_num(m["seqn"]).astype("Int64")
        for c in names[1:]:
            m[c] = to_num(m[c])
        cycle_start = int(fn.split("_")[1])
        m["cycle_start_year"] = cycle_start
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def _is_true_mask(s: pd.Series) -> pd.Series:
    if str(s.dtype) in {"bool", "boolean"}:
        return s.fillna(False).astype(bool)
    return pd.to_numeric(s, errors="coerce").eq(1)


def _disease_label(col: str) -> tuple[str, str]:
    if col in DISEASE_META:
        return DISEASE_META[col]
    m = re.fullmatch(r"(mcq\d+[a-z]?)_condition", str(col).lower())
    if m:
        code = m.group(1).upper()
        return (f"{code} condition", code)
    return (str(col).replace("_", " ").strip().title(), "")


def _fit_age_km(sub: pd.DataFrame, label: str) -> KaplanMeierFitter | None:
    kmf = KaplanMeierFitter()
    try:
        kmf.fit(
            durations=sub["end_age"],
            event_observed=sub["event"],
            entry=sub["entry_age"],
            label=label,
        )
    except StatError:
        return None
    return kmf


def _collect_disease_columns(part: pd.DataFrame) -> list[str]:
    cols = []
    for c in part.columns:
        if c in NON_DISEASE_COLUMNS:
            continue
        s = part[c]
        mask = _is_true_mask(s)
        if int(mask.sum()) <= 0:
            continue
        cols.append(c)
    return sorted(set(cols), key=lambda c: (_disease_label(c)[0].lower(), c))


def _predict_curve(kmf: KaplanMeierFitter, age_grid: np.ndarray) -> np.ndarray:
    pred = kmf.predict(pd.Series(age_grid)).astype(float).to_numpy()
    out = pred.copy()
    if np.isnan(out).all():
        out = np.ones_like(age_grid, dtype=float)
    else:
        valid = np.where(~np.isnan(out))[0]
        first = int(valid[0])
        last = int(valid[-1])
        out[:first] = 1.0
        out[last + 1 :] = out[last]
        for i in range(first + 1, len(out)):
            if np.isnan(out[i]):
                out[i] = out[i - 1]
    out = np.clip(out, 0.0, 1.0)
    out = np.minimum.accumulate(out)
    return out


def _smooth_curve(age_grid: np.ndarray, curve: np.ndarray, smooth_factor: float) -> np.ndarray:
    # Smooth KM step curve into functional representation for fPCA-like analysis.
    s = max(float(smooth_factor), 0.0) * len(age_grid)
    spl = UnivariateSpline(age_grid, curve, s=s, k=3)
    out = spl(age_grid)
    out = np.clip(out, 0.0, 1.0)
    out[0] = 1.0
    out = np.minimum.accumulate(out)
    return out


def _plot_mean_function(age_grid: np.ndarray, raw_curves: np.ndarray, smooth_curves: np.ndarray, out_path: Path) -> None:
    mu_raw = raw_curves.mean(axis=0)
    mu_sm = smooth_curves.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=170)
    ax.plot(age_grid, mu_raw, color="#64748b", linewidth=2.0, label="Mean KM (raw step)")
    ax.plot(age_grid, mu_sm, color="#0f766e", linewidth=2.8, label="Mean function (smoothed)")
    ax.set_title("Mean survival function across disease cohorts")
    ax.set_xlabel("Age")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_scree(explained: np.ndarray, out_path: Path) -> None:
    pcs = np.arange(1, len(explained) + 1)
    cum = np.cumsum(explained)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    ax.bar(pcs, explained, color="#334155", alpha=0.75, label="Per-component")
    ax.plot(pcs, cum, color="#dc2626", marker="o", linewidth=2, label="Cumulative")
    ax.set_title("fPCA explained variance")
    ax.set_xlabel("Functional principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_eigenfunctions(age_grid: np.ndarray, pca: PCA, out_path: Path, n_show: int = 4) -> None:
    n_show = min(n_show, pca.components_.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.4 * n_show), dpi=170, sharex=True)
    axes = np.array(axes).reshape(-1)
    for i in range(n_show):
        axes[i].plot(age_grid, pca.components_[i], color="#0f766e", linewidth=2.2)
        axes[i].axhline(0, color="#64748b", linewidth=1)
        axes[i].set_ylabel(f"FPC{i+1}")
        axes[i].set_title(f"FPC{i+1} (explained {pca.explained_variance_ratio_[i]*100:.1f}%)")
        axes[i].grid(alpha=0.2)
    axes[-1].set_xlabel("Age")
    fig.suptitle("Functional principal components (eigenfunctions)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_modes_of_variation(age_grid: np.ndarray, mu: np.ndarray, pca: PCA, out_path: Path) -> None:
    n_show = min(2, pca.components_.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(6.2 * n_show, 4.7), dpi=170, sharey=True)
    axes = np.array(axes).reshape(-1)
    for i in range(n_show):
        sd_score = float(np.sqrt(pca.explained_variance_[i]))
        phi = pca.components_[i]
        low = np.clip(mu - 2.0 * sd_score * phi, 0, 1)
        high = np.clip(mu + 2.0 * sd_score * phi, 0, 1)
        low = np.minimum.accumulate(low)
        high = np.minimum.accumulate(high)

        ax = axes[i]
        ax.plot(age_grid, mu, color="#0f172a", linewidth=2.4, label="Mean")
        ax.plot(age_grid, low, color="#dc2626", linewidth=2.0, label="Mean - 2 SD(FPC score)")
        ax.plot(age_grid, high, color="#2563eb", linewidth=2.0, label="Mean + 2 SD(FPC score)")
        ax.set_title(f"Mode of variation: FPC{i+1}")
        ax.set_xlabel("Age")
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("Survival")
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_silhouette_vs_k(scores_2d: np.ndarray, k_min: int, k_max: int, seed: int, out_path: Path) -> tuple[int, pd.DataFrame]:
    rows = []
    best_k = None
    best_s = -np.inf
    for k in range(k_min, k_max + 1):
        if k >= len(scores_2d):
            continue
        km = KMeans(n_clusters=k, n_init=60, random_state=seed)
        labels = km.fit_predict(scores_2d)
        if len(np.unique(labels)) < 2:
            continue
        s = float(silhouette_score(scores_2d, labels, metric="euclidean"))
        rows.append({"k": k, "silhouette": s})
        if s > best_s:
            best_s = s
            best_k = k
    if best_k is None:
        raise RuntimeError("Unable to choose k from silhouette")

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=170)
    ax.plot(df["k"], df["silhouette"], marker="o", linewidth=2, color="#0f766e")
    ax.axvline(best_k, color="#dc2626", linestyle="--", linewidth=1.6, label=f"best k={best_k}")
    ax.set_title("KMeans on (FPC1,FPC2): silhouette vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return best_k, df


def _plot_scores_scatter(scores: np.ndarray, labels: np.ndarray, meta: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=170)
    cmap = plt.get_cmap("tab10")
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        ax.scatter(scores[idx, 0], scores[idx, 1], s=85, color=cmap(int(c) % cmap.N), alpha=0.88, label=f"Cluster {int(c)+1}")
    for i, row in meta.iterrows():
        txt = str(row["cohort"]).split(" (")[0]
        ax.text(scores[i, 0] + 0.02, scores[i, 1] + 0.02, txt, fontsize=8)
    ax.axhline(0, color="#94a3b8", linewidth=1)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set_title("Disease cohorts on first two functional PC axes")
    ax.set_xlabel("FPC1 score")
    ax.set_ylabel("FPC2 score")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_cluster_overlays(age_grid: np.ndarray, raw_curves: np.ndarray, smooth_curves: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    k = len(np.unique(labels))
    fig, axes = plt.subplots(k, 1, figsize=(10, 2.9 * k), dpi=170, sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ci, c in enumerate(sorted(np.unique(labels))):
        ax = axes[ci]
        idx = np.where(labels == c)[0]
        for j in idx:
            ax.plot(age_grid, raw_curves[j], color="#94a3b8", alpha=0.2, linewidth=0.9)
            ax.plot(age_grid, smooth_curves[j], color="#0ea5e9", alpha=0.15, linewidth=1.0)
        med = np.median(smooth_curves[idx], axis=0)
        ax.plot(age_grid, med, color="#0f172a", linewidth=2.6, label=f"Cluster {int(c)+1} median (n={len(idx)})")
        ax.set_ylabel("Survival")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Age")
    fig.suptitle("Cluster-wise disease KM shape overlays (raw + smoothed)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_recon_examples(age_grid: np.ndarray, smooth_curves: np.ndarray, recon2: np.ndarray, scores: np.ndarray, meta: pd.DataFrame, out_path: Path) -> None:
    # Pick extremes on FPC1/FPC2 to illustrate reconstruction.
    idxs = []
    idxs.append(int(np.argmin(scores[:, 0])))
    idxs.append(int(np.argmax(scores[:, 0])))
    idxs.append(int(np.argmin(scores[:, 1])))
    idxs.append(int(np.argmax(scores[:, 1])))
    idxs = list(dict.fromkeys(idxs))[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=170, sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idxs):
        ax.plot(age_grid, smooth_curves[i], color="#0f172a", linewidth=2.2, label="Smoothed KM")
        ax.plot(age_grid, recon2[i], color="#dc2626", linewidth=2.0, linestyle="--", label="Reconstruction (mean + FPC1 + FPC2)")
        ax.set_title(str(meta.loc[i, "cohort"]).split(" (")[0])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Age")
        ax.set_ylabel("Survival")
    fig.suptitle("fPCA reconstruction examples using first two components", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default="data/processed/participant_health_flags.parquet")
    ap.add_argument("--mortality-dir", default="data/raw/mortality")
    ap.add_argument("--out-dir", default="output/fPCA")
    ap.add_argument("--min-disease-n", type=int, default=100)
    ap.add_argument("--age-min", type=float, default=20.0)
    ap.add_argument("--age-max", type=float, default=100.0)
    ap.add_argument("--age-step", type=float, default=1.0)
    ap.add_argument("--smooth-factor", type=float, default=0.35)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    participants = pd.read_parquet(args.participants).copy()
    participants = participants[participants["age_years"] >= 20].copy()

    mort_dir = Path(args.mortality_dir)
    download_mortality_files(mort_dir)
    mortality = load_mortality(mort_dir)

    df = participants.merge(mortality, on=["seqn", "cycle_start_year"], how="left")
    df = df[df["eligstat"] == 1].copy()
    df["time_months"] = df["permth_int"].where(df["permth_int"].notna(), df["permth_exm"])
    df["event"] = (df["mortstat"] == 1).astype(int)
    df = df[df["time_months"].notna() & df["age_years"].notna()].copy()
    df["entry_age"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["end_age"] = df["entry_age"] + pd.to_numeric(df["time_months"], errors="coerce") / 12.0
    df = df[df["entry_age"].notna() & df["end_age"].notna()].copy()
    df = df[df["end_age"] > df["entry_age"]].copy()

    disease_cols = _collect_disease_columns(participants)
    age_grid = np.arange(float(args.age_min), float(args.age_max) + 1e-9, float(args.age_step))

    rows = []
    raw_curves = []
    for col in disease_cols:
        mask = _is_true_mask(df[col])
        sub = df.loc[mask].copy()
        if sub.empty or len(sub) < int(args.min_disease_n):
            continue
        label_text, code_text = _disease_label(col)
        display = f"{label_text} ({code_text})" if code_text else label_text
        kmf = _fit_age_km(sub, display)
        if kmf is None:
            continue
        raw_curves.append(_predict_curve(kmf, age_grid))
        rows.append(
            {
                "cohort_key": col,
                "cohort": display,
                "code": code_text,
                "n": int(len(sub)),
                "deaths": int(sub["event"].sum()),
            }
        )

    if len(rows) < 4:
        raise RuntimeError("Too few disease curves available for fPCA")

    meta = pd.DataFrame(rows)
    raw_curves_arr = np.vstack(raw_curves)
    smooth_curves_arr = np.vstack([_smooth_curve(age_grid, c, smooth_factor=float(args.smooth_factor)) for c in raw_curves_arr])

    mu = smooth_curves_arr.mean(axis=0)
    X_centered = smooth_curves_arr - mu
    pca = PCA(n_components=min(X_centered.shape[0] - 1, X_centered.shape[1]))
    scores = pca.fit_transform(X_centered)

    best_k, sil_df = _plot_silhouette_vs_k(
        scores[:, :2],
        k_min=int(args.k_min),
        k_max=min(int(args.k_max), len(meta) - 1),
        seed=int(args.seed),
        out_path=out_dir / "silhouette_vs_k_fpca_scores.png",
    )

    km = KMeans(n_clusters=best_k, n_init=80, random_state=int(args.seed))
    cluster_labels = km.fit_predict(scores[:, :2])

    var_df = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained": np.cumsum(pca.explained_variance_ratio_),
            "eigenvalue": pca.explained_variance_,
        }
    )

    score_df = meta.copy()
    for i in range(min(6, scores.shape[1])):
        score_df[f"FPC{i+1}_score"] = scores[:, i]
    score_df["cluster_kmeans_fpca2d"] = cluster_labels + 1
    score_df.to_csv(out_dir / "fpca_scores_clusters.csv", index=False)

    var_df.to_csv(out_dir / "fpca_explained_variance.csv", index=False)
    sil_df.to_csv(out_dir / "fpca_k_selection_silhouette.csv", index=False)
    pd.DataFrame(raw_curves_arr, columns=[f"age_{a:g}" for a in age_grid]).assign(cohort_key=meta["cohort_key"]).to_csv(
        out_dir / "km_curves_raw_grid.csv", index=False
    )
    pd.DataFrame(smooth_curves_arr, columns=[f"age_{a:g}" for a in age_grid]).assign(cohort_key=meta["cohort_key"]).to_csv(
        out_dir / "km_curves_smoothed_grid.csv", index=False
    )
    eig_df = pd.DataFrame({"age": age_grid})
    for i in range(min(6, pca.components_.shape[0])):
        eig_df[f"FPC{i+1}_eigenfunction"] = pca.components_[i]
    eig_df.to_csv(out_dir / "fpca_eigenfunctions.csv", index=False)

    _plot_mean_function(age_grid, raw_curves_arr, smooth_curves_arr, out_dir / "mean_function.png")
    _plot_scree(pca.explained_variance_ratio_, out_dir / "scree_explained_variance.png")
    _plot_eigenfunctions(age_grid, pca, out_dir / "eigenfunctions_top_components.png", n_show=4)
    _plot_modes_of_variation(age_grid, mu, pca, out_dir / "modes_of_variation_pc1_pc2.png")
    _plot_scores_scatter(scores[:, :2], cluster_labels, meta, out_dir / "scores_scatter_pc1_pc2_clusters.png")
    _plot_cluster_overlays(age_grid, raw_curves_arr, smooth_curves_arr, cluster_labels, out_dir / "cluster_overlays_fpca2d_kmeans.png")

    recon2 = mu + scores[:, [0]] * pca.components_[0] + scores[:, [1]] * pca.components_[1]
    recon2 = np.clip(recon2, 0, 1)
    recon2 = np.minimum.accumulate(recon2, axis=1)
    _plot_recon_examples(age_grid, smooth_curves_arr, recon2, scores[:, :2], meta, out_dir / "reconstruction_examples_pc1_pc2.png")

    # Nearest neighbors in FPCA score space.
    Z = scores[:, :2]
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    nn_rows = []
    for i, row in meta.iterrows():
        ord_i = np.argsort(d[i])
        nbrs = [j for j in ord_i if j != i][:3]
        for rank, j in enumerate(nbrs, start=1):
            nn_rows.append(
                {
                    "cohort_key": row["cohort_key"],
                    "cohort": row["cohort"],
                    "neighbor_rank": rank,
                    "neighbor_cohort_key": meta.loc[j, "cohort_key"],
                    "neighbor_cohort": meta.loc[j, "cohort"],
                    "fpca2d_distance": float(d[i, j]),
                }
            )
    pd.DataFrame(nn_rows).to_csv(out_dir / "nearest_neighbors_fpca2d.csv", index=False)

    var1 = float(var_df.loc[var_df["component"] == 1, "explained_variance_ratio"].iloc[0])
    var2 = float(var_df.loc[var_df["component"] == 2, "explained_variance_ratio"].iloc[0])
    var12 = var1 + var2

    summary_txt = (
        "fPCA summary\n"
        f"disease cohorts analyzed: {len(meta)}\n"
        f"age grid: {args.age_min}..{args.age_max} step {args.age_step}\n"
        f"smoothing factor: {args.smooth_factor}\n"
        f"best k (kmeans on FPC1/FPC2): {best_k}\n"
        f"FPC1 variance explained: {var1:.4f} ({var1*100:.2f}%)\n"
        f"FPC2 variance explained: {var2:.4f} ({var2*100:.2f}%)\n"
        f"FPC1+FPC2 cumulative explained: {var12:.4f} ({var12*100:.2f}%)\n"
    )
    (out_dir / "summary.txt").write_text(summary_txt)

    print(f"Wrote fPCA outputs to: {out_dir}")
    print(f"Disease curves analyzed: {len(meta)}")
    print(f"Best K on (FPC1,FPC2): {best_k}")
    print(f"FPC1 explained variance: {var1:.4f}")
    print(f"FPC2 explained variance: {var2:.4f}")
    print(f"FPC1+FPC2 cumulative explained: {var12:.4f}")


if __name__ == "__main__":
    main()
