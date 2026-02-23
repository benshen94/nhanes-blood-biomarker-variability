#!/usr/bin/env python3
"""Cluster age-timescale NHANES disease KM curve shapes with multiple methods."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatError
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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


def _normalize_shape(curves: np.ndarray) -> np.ndarray:
    start = curves[:, [0]]
    end = curves[:, [-1]]
    drop = np.maximum(start - end, 1e-6)
    return (curves - end) / drop


def _derivative_shape(curves: np.ndarray) -> np.ndarray:
    d = -np.diff(curves, axis=1)
    row_sum = np.maximum(d.sum(axis=1, keepdims=True), 1e-8)
    return d / row_sum


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _pairwise_dtw(curves: np.ndarray) -> np.ndarray:
    n = curves.shape[0]
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            v = _dtw_distance(curves[i], curves[j])
            d[i, j] = v
            d[j, i] = v
    return d


def _choose_k_precomputed(dist_mat: np.ndarray, labels_fn, k_min: int, k_max: int) -> tuple[np.ndarray, int, float]:
    best_labels = None
    best_k = -1
    best_score = -np.inf
    n = dist_mat.shape[0]
    for k in range(k_min, min(k_max, n - 1) + 1):
        labels = labels_fn(k)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(dist_mat, labels, metric="precomputed")
        if score > best_score:
            best_score = float(score)
            best_k = int(k)
            best_labels = labels.copy()
    if best_labels is None:
        raise RuntimeError("No valid clustering found for precomputed metric")
    return best_labels, best_k, best_score


def _choose_k_feature(X: np.ndarray, labels_fn, k_min: int, k_max: int) -> tuple[np.ndarray, int, float]:
    best_labels = None
    best_k = -1
    best_score = -np.inf
    n = X.shape[0]
    for k in range(k_min, min(k_max, n - 1) + 1):
        labels = labels_fn(k)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="euclidean")
        if score > best_score:
            best_score = float(score)
            best_k = int(k)
            best_labels = labels.copy()
    if best_labels is None:
        raise RuntimeError("No valid clustering found for feature metric")
    return best_labels, best_k, best_score


def _pam_kmedoids(dist: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 24, max_iter: int = 120) -> np.ndarray:
    n = dist.shape[0]
    best_cost = np.inf
    best_labels = None
    idx = np.arange(n)

    for _ in range(n_init):
        medoids = rng.choice(idx, size=k, replace=False)
        for _it in range(max_iter):
            d_to_medoids = dist[:, medoids]
            labels = np.argmin(d_to_medoids, axis=1)
            cost = d_to_medoids[np.arange(n), labels].sum()
            improved = False
            for mi in range(k):
                m_old = medoids[mi]
                for cand in idx:
                    if cand in medoids:
                        continue
                    trial = medoids.copy()
                    trial[mi] = cand
                    d_trial = dist[:, trial]
                    labels_trial = np.argmin(d_trial, axis=1)
                    cost_trial = d_trial[np.arange(n), labels_trial].sum()
                    if cost_trial + 1e-10 < cost:
                        medoids = trial
                        labels = labels_trial
                        cost = cost_trial
                        improved = True
            if not improved:
                break
        if cost < best_cost:
            best_cost = float(cost)
            best_labels = labels.copy()

    if best_labels is None:
        raise RuntimeError("k-medoids failed")
    return best_labels


def _save_heatmap(dist: np.ndarray, labels: list[str], out_path: Path, title: str, order: np.ndarray | None = None) -> None:
    if order is None:
        order = np.arange(len(labels))
    d = dist[np.ix_(order, order)]
    lbl = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(11, 9), dpi=170)
    sns.heatmap(d, xticklabels=lbl, yticklabels=lbl, cmap="mako", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_method_overlays(method_results: dict, curves: np.ndarray, ages: np.ndarray, names: list[str], out_path: Path) -> None:
    n_methods = len(method_results)
    ncols = 2
    nrows = int(math.ceil(n_methods / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.4 * nrows), dpi=170, sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, (method, info) in enumerate(method_results.items()):
        ax = axes[i]
        labels = np.array(info["labels"])
        k = int(info["k"])
        cmap = plt.get_cmap("tab10")
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            color = cmap(c % cmap.N)
            for j in idx:
                ax.plot(ages, curves[j], color=color, alpha=0.22, linewidth=1.1)
            med = np.median(curves[idx], axis=0)
            ax.plot(ages, med, color=color, linewidth=2.8, label=f"C{c + 1} (n={len(idx)})")
        ax.set_title(f"{method} | k={k}, silhouette={info['silhouette']:.3f}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Survival")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8, ncol=2)

    for j in range(n_methods, len(axes)):
        axes[j].axis("off")

    fig.suptitle("KM Curve-Shape Clustering Overlays (Disease cohorts)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_consensus(consensus: np.ndarray, labels: list[str], out_path: Path, order: np.ndarray) -> None:
    c = consensus[np.ix_(order, order)]
    lbl = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(11, 9), dpi=170)
    sns.heatmap(c, xticklabels=lbl, yticklabels=lbl, cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Consensus similarity (fraction of methods that co-cluster each pair)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default="data/processed/participant_health_flags.parquet")
    ap.add_argument("--mortality-dir", default="data/raw/mortality")
    ap.add_argument("--out-dir", default="output/km_shape_clustering")
    ap.add_argument("--min-disease-n", type=int, default=100)
    ap.add_argument("--age-min", type=float, default=20.0)
    ap.add_argument("--age-max", type=float, default=100.0)
    ap.add_argument("--age-step", type=float, default=1.0)
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
    curve_list = []
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
        curve = _predict_curve(kmf, age_grid)
        curve_list.append(curve)
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
        raise RuntimeError("Too few disease curves available for clustering")

    meta = pd.DataFrame(rows)
    curves = np.vstack(curve_list)
    curves_shape = _normalize_shape(curves)
    curves_deriv = _derivative_shape(curves)

    meta.to_csv(out_dir / "disease_curve_metadata.csv", index=False)
    pd.DataFrame(curves, columns=[f"age_{a:g}" for a in age_grid]).assign(cohort_key=meta["cohort_key"]).to_csv(
        out_dir / "km_curve_matrix.csv", index=False
    )

    dist_mats = {
        "cosine_raw": squareform(pdist(curves, metric="cosine")),
        "euclidean_raw": squareform(pdist(curves, metric="euclidean")),
        "correlation_raw": squareform(pdist(curves, metric="correlation")),
        "cosine_shape": squareform(pdist(curves_shape, metric="cosine")),
        "euclidean_derivative": squareform(pdist(curves_deriv, metric="euclidean")),
        "dtw_shape": _pairwise_dtw(curves_shape),
    }

    names = meta["cohort"].tolist()

    link_cos = linkage(squareform(dist_mats["cosine_raw"], checks=False), method="average")
    order = np.array(dendrogram(link_cos, no_plot=True)["leaves"], dtype=int)

    for metric_name, dist in dist_mats.items():
        df_dist = pd.DataFrame(dist, index=meta["cohort_key"], columns=meta["cohort_key"])
        df_dist.to_csv(out_dir / f"pairwise_distance_{metric_name}.csv")
        _save_heatmap(dist, names, out_dir / f"heatmap_{metric_name}.png", f"Pairwise distance heatmap: {metric_name}", order=order)

    # Convenience table: nearest neighbors by cosine distance on raw KM curves.
    cos = dist_mats["cosine_raw"]
    nn_rows = []
    for i, row in meta.iterrows():
        order_i = np.argsort(cos[i])
        nbrs = [j for j in order_i if j != i][:3]
        for rank, j in enumerate(nbrs, start=1):
            nn_rows.append(
                {
                    "cohort_key": row["cohort_key"],
                    "cohort": row["cohort"],
                    "neighbor_rank": rank,
                    "neighbor_cohort_key": meta.loc[j, "cohort_key"],
                    "neighbor_cohort": meta.loc[j, "cohort"],
                    "cosine_distance": float(cos[i, j]),
                }
            )
    pd.DataFrame(nn_rows).to_csv(out_dir / "nearest_neighbors_cosine_raw.csv", index=False)

    fig_d, ax_d = plt.subplots(figsize=(12, 5), dpi=170)
    dendrogram(link_cos, labels=names, leaf_rotation=90, leaf_font_size=8, ax=ax_d)
    ax_d.set_title("Hierarchical dendrogram (average linkage, cosine distance on raw KM curves)")
    fig_d.tight_layout()
    fig_d.savefig(out_dir / "dendrogram_cosine_raw.png")
    plt.close(fig_d)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=args.seed, normalized_stress="auto")
    emb = mds.fit_transform(dist_mats["cosine_raw"])
    fig_m, ax_m = plt.subplots(figsize=(9, 7), dpi=170)
    ax_m.scatter(emb[:, 0], emb[:, 1], s=90, color="#0f766e", alpha=0.86)
    for i, txt in enumerate(names):
        ax_m.text(emb[i, 0] + 0.01, emb[i, 1] + 0.01, txt.split(" (")[0], fontsize=8)
    ax_m.set_title("MDS map of KM curve shapes (cosine distance)")
    ax_m.set_xlabel("MDS-1")
    ax_m.set_ylabel("MDS-2")
    ax_m.grid(alpha=0.25)
    fig_m.tight_layout()
    fig_m.savefig(out_dir / "mds_cosine_raw.png")
    plt.close(fig_m)

    n = len(meta)
    k_max = min(int(args.k_max), max(2, n - 1))
    k_min = min(int(args.k_min), k_max)
    rng = np.random.default_rng(int(args.seed))

    method_results = {}

    def hier_cos_labels(k: int) -> np.ndarray:
        return fcluster(link_cos, t=k, criterion="maxclust") - 1

    labels, k, sil = _choose_k_precomputed(dist_mats["cosine_raw"], hier_cos_labels, k_min, k_max)
    method_results["hierarchical_avg_cosine"] = {"labels": labels, "k": k, "silhouette": sil}

    link_dtw = linkage(squareform(dist_mats["dtw_shape"], checks=False), method="average")

    def hier_dtw_labels(k: int) -> np.ndarray:
        return fcluster(link_dtw, t=k, criterion="maxclust") - 1

    labels, k, sil = _choose_k_precomputed(dist_mats["dtw_shape"], hier_dtw_labels, k_min, k_max)
    method_results["hierarchical_avg_dtw"] = {"labels": labels, "k": k, "silhouette": sil}

    def pam_labels(k: int) -> np.ndarray:
        return _pam_kmedoids(dist_mats["cosine_raw"], k=k, rng=rng)

    labels, k, sil = _choose_k_precomputed(dist_mats["cosine_raw"], pam_labels, k_min, k_max)
    method_results["kmedoids_cosine"] = {"labels": labels, "k": k, "silhouette": sil}

    X_scaled = StandardScaler().fit_transform(curves)

    def kmeans_labels(k: int) -> np.ndarray:
        return KMeans(n_clusters=k, n_init=20, random_state=int(args.seed)).fit_predict(X_scaled)

    labels, k, sil = _choose_k_feature(X_scaled, kmeans_labels, k_min, k_max)
    method_results["kmeans_euclidean"] = {"labels": labels, "k": k, "silhouette": sil}

    cos = dist_mats["cosine_raw"]
    sigma = float(np.median(cos[cos > 0])) if np.any(cos > 0) else 1.0
    affinity = np.exp(-np.square(cos) / (2 * sigma * sigma + 1e-12))

    def spec_labels(k: int) -> np.ndarray:
        return SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=int(args.seed),
            n_init=25,
            assign_labels="kmeans",
        ).fit_predict(affinity)

    labels, k, sil = _choose_k_precomputed(cos, spec_labels, k_min, k_max)
    method_results["spectral_cosine"] = {"labels": labels, "k": k, "silhouette": sil}

    agg = AgglomerativeClustering(metric="euclidean", linkage="ward")

    def ward_labels(k: int) -> np.ndarray:
        model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        return model.fit_predict(X_scaled)

    labels, k, sil = _choose_k_feature(X_scaled, ward_labels, k_min, k_max)
    method_results["agglomerative_ward"] = {"labels": labels, "k": k, "silhouette": sil}

    method_rows = []
    assign_rows = []
    for method, info in method_results.items():
        method_rows.append({"method": method, "best_k": int(info["k"]), "silhouette": float(info["silhouette"])})
        labels = np.array(info["labels"])
        for i, row in meta.iterrows():
            assign_rows.append(
                {
                    "method": method,
                    "cohort_key": row["cohort_key"],
                    "cohort": row["cohort"],
                    "n": int(row["n"]),
                    "deaths": int(row["deaths"]),
                    "cluster": int(labels[i]) + 1,
                }
            )

    method_df = pd.DataFrame(method_rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    assign_df = pd.DataFrame(assign_rows)
    method_df.to_csv(out_dir / "cluster_method_summary.csv", index=False)
    assign_df.to_csv(out_dir / "cluster_assignments.csv", index=False)

    fig_s, ax_s = plt.subplots(figsize=(9, 4.6), dpi=170)
    mplot = method_df.sort_values("silhouette", ascending=True)
    ax_s.barh(mplot["method"], mplot["silhouette"], color="#0f766e")
    ax_s.set_title("Clustering method comparison (best silhouette)")
    ax_s.set_xlabel("Silhouette score")
    ax_s.grid(axis="x", alpha=0.2)
    fig_s.tight_layout()
    fig_s.savefig(out_dir / "cluster_method_silhouette.png")
    plt.close(fig_s)

    _plot_method_overlays(method_results, curves, age_grid, names, out_dir / "cluster_overlays_by_method.png")

    # Consensus (fraction of methods that place pair in same cluster)
    mnames = list(method_results.keys())
    n_methods = len(mnames)
    consensus = np.zeros((n, n), dtype=float)
    for m in mnames:
        labels = np.array(method_results[m]["labels"])
        same = (labels[:, None] == labels[None, :]).astype(float)
        consensus += same
    consensus /= float(n_methods)
    pd.DataFrame(consensus, index=meta["cohort_key"], columns=meta["cohort_key"]).to_csv(out_dir / "consensus_similarity_matrix.csv")
    _plot_consensus(consensus, names, out_dir / "consensus_similarity_heatmap.png", order=order)

    print(f"Wrote clustering outputs to: {out_dir}")
    print(f"Disease curves clustered: {len(meta)}")
    print("Methods run:", ", ".join(method_df["method"].tolist()))


if __name__ == "__main__":
    main()
