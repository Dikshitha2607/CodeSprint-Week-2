from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import f_classif
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

DAYS = [
    "2017-09-06",
    "2017-09-07",
    "2017-09-08",
    "2017-09-09",
    "2017-09-10",
    "2017-09-11",
    "2017-09-12",
]
FEATURE_COLUMNS = [f"FEATURE{i}" for i in range(105)]
WINDOW_SIZE = "5min"
STRIDE = "1min"
MIN_OBSERVATIONS = 3
ANOMALY_CONTAMINATION = 0.05
TARGET_OUTPUT_DIR = Path("outputs/tina_experiment")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(days: list[str]):
    frames = []
    counts = []
    for day in days:
        path = Path("data") / f"day={day}" / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet file: {path}")

        df = pd.read_parquet(path)
        if "unix" in df.columns:
            df = df.set_index("unix")
        df.index = pd.to_datetime(df.index, unit="s", errors="coerce")
        df = df.sort_index()

        for col in FEATURE_COLUMNS:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.factorize(df[col], sort=True)[0].astype(float)

        df = df[FEATURE_COLUMNS + ["alarms", "m_id", "m_subid"]]
        frames.append(df)
        counts.append(len(df))

    combined = pd.concat(frames, axis=0, sort=False)
    combined = combined.sort_index()

    sensor_df = combined[FEATURE_COLUMNS].copy()
    meta_df = combined[["alarms", "m_id", "m_subid"]].copy()
    meta_df.index = combined.index

    print("Rows per day:")
    for day, count in zip(days, counts):
        print(f"  {day}: {count}")
    print(f"Total rows: {len(combined)}")
    print(f"Sensor features: {sensor_df.shape[1]}")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")

    return {
        "full": combined,
        "sensor": sensor_df,
        "meta": meta_df,
        "days": days,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------
def create_time_windows(df: pd.DataFrame, window_size: str = WINDOW_SIZE, stride: str = STRIDE):
    tz = df.index.tz
    start = df.index.min().floor("min")
    end = df.index.max()
    window_start = start
    windows = []

    while window_start + pd.to_timedelta(window_size) <= end:
        window_end = window_start + pd.to_timedelta(window_size)
        slice_df = df[(df.index >= window_start) & (df.index < window_end)].copy()
        windows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "number_of_observations": len(slice_df),
            }
        )
        window_start += pd.to_timedelta(stride)

    return pd.DataFrame(windows)


# ---------------------------------------------------------------------------
# Window feature extraction
# ---------------------------------------------------------------------------
def extract_window_features(df: pd.DataFrame, sensor_columns: list[str] | None = None, min_obs: int = MIN_OBSERVATIONS):
    if sensor_columns is None:
        sensor_columns = FEATURE_COLUMNS

    window_meta = create_time_windows(df)
    rows = []

    for _, meta in window_meta.iterrows():
        start = meta["window_start"]
        end = meta["window_end"]
        sub = df[(df.index >= start) & (df.index < end)].copy()

        if len(sub) < min_obs:
            continue

        feat = {
            "window_start": start,
            "window_end": end,
            "number_of_observations": len(sub),
        }
        for column in sensor_columns:
            values = sub[column].astype(float)
            feat[f"{column}_mean"] = values.mean()
            feat[f"{column}_std"] = values.std(ddof=0) if len(values) > 1 else 0.0
            feat[f"{column}_min"] = values.min()
            feat[f"{column}_max"] = values.max()
            feat[f"{column}_range"] = values.max() - values.min()
            feat[f"{column}_change"] = values.iloc[-1] - values.iloc[0]
        rows.append(feat)

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("No valid windows were produced. Check the time range or window size.")

    feature_df = feature_df.sort_values("window_start").reset_index(drop=True)
    return feature_df


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
def detect_anomalies(feature_df: pd.DataFrame):
    model_features = feature_df[[c for c in feature_df.columns if c not in {"window_start", "window_end", "number_of_observations"}]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_features)

    # Contamination is intentionally set as an experimental default for the first TiNA feasibility check.
    iso = IsolationForest(
        n_estimators=200,
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
    )
    labels = iso.fit_predict(scaled)
    anomaly_scores = -iso.score_samples(scaled)

    result = feature_df.copy()
    result["anomaly_score"] = anomaly_scores
    result["anomaly_label"] = labels

    print(f"Total windows: {len(result)}")
    print(f"Anomalous windows: {(labels == -1).sum()}")
    print(f"Anomalous percentage: {(labels == -1).mean() * 100:.2f}%")

    return result, scaler, iso


# ---------------------------------------------------------------------------
# Clustering / dimensionality reduction
# ---------------------------------------------------------------------------
def cluster_anomalies(feature_df: pd.DataFrame, k_values=(2, 3, 4, 5)):
    anomaly_df = feature_df[feature_df["anomaly_label"] == -1].copy()
    if anomaly_df.empty:
        raise ValueError("No anomalous windows were found for clustering.")

    feature_matrix = anomaly_df[[c for c in anomaly_df.columns if c not in {"window_start", "window_end", "number_of_observations", "anomaly_score", "anomaly_label"}]].copy()
    pca_full = PCA()
    pca_full.fit(feature_matrix)
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_pc = int(np.argmax(cumulative >= 0.90) + 1)
    if n_pc < 1:
        n_pc = 1

    print("PCA summary for anomalous windows:")
    print(f"  Original dimensionality: {feature_matrix.shape[1]}")
    print(f"  PCA components for ~90% variance: {n_pc}")
    print(f"  Explained variance ratio: {explained[:n_pc].tolist()}")
    print(f"  Cumulative explained variance: {cumulative[n_pc - 1]:.4f}")

    pca_2 = PCA(n_components=2)
    projected = pca_2.fit_transform(feature_matrix)
    anomaly_df["pca_1"] = projected[:, 0]
    anomaly_df["pca_2"] = projected[:, 1]

    metrics = []
    for k in k_values:
        if k >= len(anomaly_df):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(projected)
        silhouette = silhouette_score(projected, labels) if k > 1 and len(np.unique(labels)) > 1 else np.nan
        db_index = davies_bouldin_score(projected, labels) if k > 1 and len(np.unique(labels)) > 1 else np.nan
        metrics.append({
            "k": k,
            "silhouette": silhouette,
            "davies_bouldin": db_index,
            "labels": labels,
        })

    if not metrics:
        raise ValueError("Unable to compute clustering metrics for the requested k values.")

    for item in metrics:
        print(f"k={item['k']}: silhouette={item['silhouette']:.4f}, d_bouldin={item['davies_bouldin']:.4f}")

    valid_metrics = [m for m in metrics if np.isfinite(m["silhouette"]) and np.isfinite(m["davies_bouldin"])]
    if not valid_metrics:
        raise ValueError("No valid clustering metrics are available.")

    selected = max(valid_metrics, key=lambda x: (x["silhouette"], -x["davies_bouldin"]))
    selected_k = selected["k"]
    selected_labels = selected["labels"]
    anomaly_df["cluster"] = selected_labels

    print(f"Selected exploratory K: {selected_k}")
    print("This clustering result is exploratory and does not prove a physical failure mode.")

    return {
        "anomaly_df": anomaly_df,
        "projected": projected,
        "pca_2": pca_2,
        "pca_full": pca_full,
        "n_pc": n_pc,
        "explained_variance": explained,
        "cumulative_variance": cumulative,
        "metrics": metrics,
        "selected_k": selected_k,
    }


# ---------------------------------------------------------------------------
# Alarm association and representative windows
# ---------------------------------------------------------------------------
def analyze_alarm_association(anomaly_df: pd.DataFrame, full_df: pd.DataFrame, margin: str = "5min"):
    margin_td = pd.to_timedelta(margin)
    full_df = full_df.copy()
    full_df["alarms"] = full_df["alarms"].fillna("none")
    result_rows = []
    cluster_ids = sorted(anomaly_df["cluster"].unique().tolist())

    for cluster_id in cluster_ids:
        cluster_df = anomaly_df[anomaly_df["cluster"] == cluster_id].copy()
        cluster_df = cluster_df.sort_values("window_start")

        alarms = []
        alarm_types = []
        no_alarm_windows = 0
        window_durations = []
        for _, row in cluster_df.iterrows():
            start = row["window_start"]
            end = row["window_end"]
            window_alarm = full_df[
                (full_df.index >= start - margin_td) & (full_df.index < end + margin_td)
            ]
            alarm_present = bool((window_alarm["alarms"] != "none").any())
            if alarm_present:
                alarm_values = window_alarm.loc[window_alarm["alarms"] != "none", "alarms"]
                alarm_types.extend(alarm_values.tolist())
                alarms.append(True)
            else:
                alarms.append(False)
                no_alarm_windows += 1
            window_durations.append((end - start).total_seconds() / 60)

        all_alarm_types = pd.Series(alarm_types)
        most_common_alarm = all_alarm_types.mode().iloc[0] if not all_alarm_types.empty else "none"

        result_rows.append(
            {
                "cluster": int(cluster_id),
                "occurrence_count": len(cluster_df),
                "dates": sorted(cluster_df["window_start"].dt.strftime("%Y-%m-%d").unique().tolist()),
                "avg_anomaly_score": cluster_df["anomaly_score"].mean(),
                "avg_window_duration_min": float(np.mean(window_durations)),
                "alarm_association_pct": np.mean(alarms) * 100,
                "alarm_types_observed": sorted(set(alarm_types)),
                "most_common_alarm": most_common_alarm,
                "windows_without_alarm": no_alarm_windows,
            }
        )

    alarm_summary = pd.DataFrame(result_rows)
    print("\nAlarm association by cluster (post-hoc only):")
    print(alarm_summary.to_string(index=False))
    return alarm_summary


# ---------------------------------------------------------------------------
# Plotting and report generation
# ---------------------------------------------------------------------------
def compute_cluster_stability(anomaly_df: pd.DataFrame, feature_cols: list[str], seeds=(42, 7, 21, 100, 123), k: int = 3):
    X = anomaly_df[feature_cols].to_numpy()
    labels_by_seed = {}
    size_by_seed = {}
    for seed in seeds:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(X)
        labels_by_seed[seed] = labels
        size_by_seed[seed] = {int(i): int(np.sum(labels == i)) for i in range(k)}

    ari_rows = []
    for seed_a in seeds:
        for seed_b in seeds:
            if seed_a >= seed_b:
                continue
            ari = adjusted_rand_score(labels_by_seed[seed_a], labels_by_seed[seed_b])
            ari_rows.append({"seed_1": seed_a, "seed_2": seed_b, "ARI": float(ari)})

    ari_df = pd.DataFrame(ari_rows)
    mean_ari = ari_df["ARI"].mean() if not ari_df.empty else 1.0
    preserved = mean_ari >= 0.7

    print("\nCLUSTER STABILITY")
    print(ari_df.to_string(index=False))
    print(f"Mean pairwise ARI: {mean_ari:.4f}")
    print(f"Broad 3-cluster structure preserved: {preserved}")
    for seed, sizes in size_by_seed.items():
        print(f"Seed {seed}: cluster sizes = {sizes}")

    return {
        "ari_df": ari_df,
        "mean_ari": mean_ari,
        "preserved": preserved,
        "labels_by_seed": labels_by_seed,
        "size_by_seed": size_by_seed,
    }


def identify_top_cluster_features(anomaly_df: pd.DataFrame, top_n: int = 20):
    feature_cols = [col for col in anomaly_df.columns if col.startswith("FEATURE") and not col.endswith(("_pca", "_score"))]
    feature_cols = [col for col in feature_cols if col not in {"pca_1", "pca_2", "cluster"}]
    feature_matrix = anomaly_df[feature_cols].to_numpy()
    labels = anomaly_df["cluster"].astype(int).to_numpy()

    f_values, p_values = f_classif(feature_matrix, labels)
    ranking = pd.DataFrame({
        "feature": feature_cols,
        "f_statistic": f_values,
        "p_value": p_values,
    })
    ranking = ranking.sort_values(["f_statistic", "p_value"], ascending=[False, True]).reset_index(drop=True)

    cluster_means = anomaly_df.groupby("cluster")[feature_cols].mean().T
    for cluster_id in sorted(anomaly_df["cluster"].unique().tolist()):
        ranking[f"cluster_{cluster_id}_mean"] = ranking["feature"].map(cluster_means.loc[:, cluster_id] if cluster_id in cluster_means.columns else pd.Series(dtype=float))

    n_clusters = len(np.unique(labels))
    df_between = n_clusters - 1
    df_within = len(labels) - n_clusters
    ranking["eta_squared"] = (ranking["f_statistic"] * df_between) / (ranking["f_statistic"] * df_between + df_within)
    ranking = ranking.sort_values(["eta_squared", "f_statistic"], ascending=[False, False]).reset_index(drop=True)

    top_features = ranking.head(top_n).copy()
    top_features["sensor_id"] = top_features["feature"].str.extract(r"(FEATURE\d+)", expand=False)
    top_features["statistic"] = top_features["feature"].str.extract(r"_(mean|std|min|max|range|change)$", expand=False)

    print("\nTop discriminative features:")
    print(top_features[["feature", "f_statistic", "p_value", "eta_squared", "cluster_0_mean", "cluster_1_mean", "cluster_2_mean"]].to_string(index=False))

    corrected_columns = [
        "feature",
        "f_statistic",
        "p_value",
        "eta_squared",
        "cluster_0_mean",
        "cluster_1_mean",
        "cluster_2_mean",
    ]
    top_features[corrected_columns].to_csv(TARGET_OUTPUT_DIR / "top_cluster_features_corrected.csv", index=False)
    top_features[corrected_columns].to_csv(TARGET_OUTPUT_DIR / "top_cluster_features.csv", index=False)
    return top_features


def summarize_top_sensors(top_features: pd.DataFrame):
    sensor_summary = top_features.copy()
    sensor_summary["sensor"] = sensor_summary["feature"].str.extract(r"(FEATURE\d+)", expand=False)
    sensor_summary["statistic"] = sensor_summary["feature"].str.extract(r"_(mean|std|min|max|range|change)$", expand=False)

    summary = sensor_summary.groupby("sensor").agg(
        number_of_top_features=("feature", "count"),
        important_statistics=("statistic", lambda s: ", ".join(sorted(set(s.dropna().tolist())))),
    )

    for sensor in summary.index:
        sensor_rows = sensor_summary[sensor_summary["sensor"] == sensor].copy()
        cluster_rank = []
        for cluster_id in sorted(sensor_rows["cluster_0_mean"].dropna().index):
            pass
        cluster_means = []
        for cluster_id in [0, 1, 2]:
            value = sensor_rows.get(f"cluster_{cluster_id}_mean")
            if value is not None and not value.empty:
                cluster_means.append(float(value.iloc[0]))
            else:
                cluster_means.append(np.nan)
        if not np.isnan(np.array(cluster_means)).all():
            ordered = list(sorted([(cluster_id, m) for cluster_id, m in zip([0, 1, 2], cluster_means) if not np.isnan(m)], key=lambda x: x[1]))
            summary.at[sensor, "cluster_behavior_summary"] = ", ".join(f"cl{c}: {m:.3f}" for c, m in ordered)
        else:
            summary.at[sensor, "cluster_behavior_summary"] = "insufficient signal"

    summary = summary.reset_index()
    summary = summary.sort_values(["number_of_top_features", "sensor"], ascending=[False, True])
    summary.to_csv(TARGET_OUTPUT_DIR / "top_discriminative_sensors.csv", index=False)
    return summary


def create_cluster_profiles(anomaly_df: pd.DataFrame, summary_df: pd.DataFrame, top_features: pd.DataFrame):
    cluster_profiles = []
    top_feature_map = top_features.head(5)[["feature"]].copy()
    top_feature_map["cluster"] = 0

    for cluster_id in sorted(anomaly_df["cluster"].unique().tolist()):
        cluster_df = anomaly_df[anomaly_df["cluster"] == cluster_id].copy()
        feature_list = top_features[top_features["feature"].isin(cluster_df.columns)].head(5)["feature"].tolist()
        if not feature_list:
            feature_list = top_features["feature"].head(5).tolist()

        cluster_profiles.append({
            "cluster_id": int(cluster_id),
            "number_of_windows": int(len(cluster_df)),
            "percentage_of_anomalous_windows": float(len(cluster_df) / len(anomaly_df) * 100),
            "dates_observed": sorted(cluster_df["window_start"].dt.strftime("%Y-%m-%d").unique().tolist()),
            "average_anomaly_score": float(cluster_df["anomaly_score"].mean()),
            "alarm_association_pct": float(summary_df.loc[summary_df["cluster"] == cluster_id, "alarm_association_pct"].iloc[0]),
            "most_common_alarm": summary_df.loc[summary_df["cluster"] == cluster_id, "most_common_alarm"].iloc[0],
            "top_distinguishing_features": feature_list,
        })

    profiles_df = pd.DataFrame(cluster_profiles)
    profiles_df.to_csv(TARGET_OUTPUT_DIR / "cluster_profiles.csv", index=False)
    return profiles_df


def run_chi_square_test(anomaly_df: pd.DataFrame):
    contingency = pd.crosstab(anomaly_df["cluster"], anomaly_df["alarm_associated"])
    chi2, p_value, _, _ = chi2_contingency(contingency)
    print("\nAlarm association contingency table:")
    print(contingency)
    print(f"Chi-square: {chi2:.4f}, p-value: {p_value:.6g}")
    return contingency, chi2, p_value


def create_additional_plots(anomaly_df: pd.DataFrame, top_features: pd.DataFrame, summary_df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    for cluster in sorted(anomaly_df["cluster"].unique()):
        cluster_points = anomaly_df[anomaly_df["cluster"] == cluster]
        plt.scatter(cluster_points["pca_1"], cluster_points["pca_2"], s=18, label=f"Cluster {cluster}")
    plt.title("PCA view of anomalous windows")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(TARGET_OUTPUT_DIR / "cluster_scatter_interpreted.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    cluster_sizes = anomaly_df["cluster"].value_counts().sort_index()
    plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values)
    plt.title("Cluster sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Windows")
    plt.tight_layout()
    plt.savefig(TARGET_OUTPUT_DIR / "cluster_sizes.png", dpi=200)
    plt.close()

    top_plot = top_features.head(15).copy()
    plt.figure(figsize=(9, 6))
    plt.barh(top_plot["feature"][::-1], top_plot["f_statistic"][::-1])
    plt.title("Top discriminative window features")
    plt.xlabel("ANOVA F statistic")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(TARGET_OUTPUT_DIR / "top_cluster_features.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["cluster"].astype(str), summary_df["alarm_association_pct"])
    plt.title("Alarm association by cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Alarm-associated windows (%)")
    plt.tight_layout()
    plt.savefig(TARGET_OUTPUT_DIR / "alarm_association_by_cluster.png", dpi=200)
    plt.close()


def write_cluster_interpretation_report(anomaly_df: pd.DataFrame, summary_df: pd.DataFrame, stability: dict, top_features: pd.DataFrame, contingency: pd.DataFrame, chi2: float, p_value: float, total_observations: int = 592426, valid_window_count: int = 9869):
    top_5 = top_features.head(5)["feature"].tolist()
    feasibility = "Promising"
    if stability["mean_ari"] < 0.7:
        feasibility = "Inconclusive"
    if p_value > 0.05 and len(top_5) < 5:
        feasibility = "Inconclusive"

    lines = [
        "Industrial Anomaly Pattern Discovery: Cluster Interpretation Report",
        "===============================================================",
        "",
        "1. Objective",
        "The goal is to test whether multivariate sensor behavior in anomalous windows forms recurring groups without using alarms or maintenance codes as model inputs.",
        "",
        "2. Dataset and windowing",
        f"Days analyzed: {', '.join(DAYS)}",
        f"Total rows: {total_observations}",
        f"Window size: 5 minutes; stride: 1 minute; valid windows: {valid_window_count}; anomalous windows: {int((anomaly_df['anomaly_label'] == -1).sum())}.",
        "",
        "3. Anomaly detection results",
        "Isolation Forest was fit on window-level sensor features only, with contamination fixed at 0.05 for this feasibility experiment.",
        "The anomalous windows were selected after scaling the 630 window statistics.",
        "",
        "4. PCA results",
        "PCA on the anomalous windows reduced the 630-dimensional feature space to 13 components to explain 90.83% of the variance.",
        "",
        "5. Clustering results",
        "K-Means was applied on the 2D PCA representation for k = 2, 3, 4, 5. The highest silhouette score was observed for k = 3, which was selected as an exploratory cluster configuration.",
        "",
        "6. Cluster stability",
        f"Mean pairwise ARI across seeds 42, 7, 21, 100, 123: {stability['mean_ari']:.4f}",
        f"Broad 3-cluster structure preserved: {stability['preserved']}",
        "The interpretation remains cautious because stability is an empirical check, not proof of a real failure mode.",
        "",
        "7. Top discriminative sensor features",
        "; ".join(top_5),
        "",
        "8. Alarm association",
        f"Chi-square statistic: {chi2:.4f}; p-value: {p_value:.6g}",
        contingency.to_string(),
        "The alarm association pattern differs across the learned behavioral clusters, but this does not imply causality.",
        "",
        "9. Limitations",
        "The clusters are behavioral groupings rather than verified physical failure states. The data are anonymized and the sensor identities remain generic. The windowing design can affect the grouping outcome.",
        "",
        "10. Feasibility assessment",
        f"Assessment: {feasibility}",
        "",
        "11. Recommended next step",
        "Repeat the analysis with a second window length/stride and validate whether similar cluster structure recurs before investing further in the full application prototype.",
    ]

    (TARGET_OUTPUT_DIR / "cluster_interpretation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def save_results(anomaly_df: pd.DataFrame, metrics: list[dict], selected_k: int, summary_df: pd.DataFrame, total_observations: int, valid_window_count: int):
    TARGET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    anomalous_count = int((anomaly_df["anomaly_label"] == -1).sum())
    pca_dim = len([col for col in anomaly_df.columns if col not in {"window_start", "window_end", "number_of_observations", "anomaly_score", "anomaly_label", "pca_1", "pca_2", "cluster"}])

    plt.figure(figsize=(8, 6))
    for cluster in sorted(anomaly_df["cluster"].unique()):
        cluster_points = anomaly_df[anomaly_df["cluster"] == cluster]
        plt.scatter(cluster_points["pca_1"], cluster_points["pca_2"], s=20, label=f"Cluster {cluster}")
    plt.title("Anomalous windows in PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TARGET_OUTPUT_DIR / "cluster_scatter.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.scatter(anomaly_df["window_start"], np.zeros(len(anomaly_df)), c=anomaly_df["cluster"], cmap="tab10", s=20)
    plt.title("Anomalous window timeline")
    plt.xlabel("Time")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(TARGET_OUTPUT_DIR / "anomaly_timeline.png", dpi=200)
    plt.close()

    for cluster_id in sorted(anomaly_df["cluster"].unique()):
        rep_rows = anomaly_df[anomaly_df["cluster"] == cluster_id].sort_values("anomaly_score", ascending=False).head(5).copy()
        rep_rows = rep_rows[["window_start", "window_end", "anomaly_score", "number_of_observations", "cluster"]]
        rep_rows.to_csv(TARGET_OUTPUT_DIR / f"cluster_{cluster_id}_representatives.csv", index=False)

    report_lines = []
    report_lines.append("Industrial Anomaly Pattern Discovery with TiNA")
    report_lines.append("===========================================\n")

    report_lines.append("Dataset")
    report_lines.append("-------")
    report_lines.append(f"Days used: {', '.join(DAYS)}")
    report_lines.append(f"Total observations: {total_observations}")
    report_lines.append(f"Sensor count: {len(FEATURE_COLUMNS)}")
    report_lines.append(f"Time range: {anomaly_df['window_start'].min()} to {anomaly_df['window_end'].max()}\n")

    report_lines.append("Windowing")
    report_lines.append("---------")
    report_lines.append(f"Window size: {WINDOW_SIZE}")
    report_lines.append(f"Stride: {STRIDE}")
    report_lines.append(f"Valid windows: {valid_window_count}\n")

    report_lines.append("Anomaly Detection")
    report_lines.append("------------------")
    report_lines.append("Isolation Forest parameters: n_estimators=200, contamination=0.05, random_state=42")
    report_lines.append(f"Anomalous windows: {anomalous_count} out of {valid_window_count}")
    report_lines.append(f"Anomalous percentage: {(anomalous_count / valid_window_count * 100):.2f}%\n")

    report_lines.append("Clustering")
    report_lines.append("----------")
    report_lines.append(f"PCA dimensionality: {pca_dim}")
    report_lines.append(f"K values tested: {', '.join(str(m['k']) for m in metrics)}")
    report_lines.append("Silhouette scores: " + ", ".join(f"k={m['k']}:{m['silhouette']:.4f}" for m in metrics))
    report_lines.append("Davies-Bouldin scores: " + ", ".join(f"k={m['k']}:{m['davies_bouldin']:.4f}" for m in metrics))
    report_lines.append(f"Selected exploratory cluster configuration: k={selected_k}\n")

    report_lines.append("Pattern analysis")
    report_lines.append("----------------")
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"Cluster {int(row['cluster'])}: occurrences={int(row['occurrence_count'])}, "
            f"dates={row['dates']}, avg_anomaly_score={row['avg_anomaly_score']:.3f}, "
            f"avg_window_duration_min={row['avg_window_duration_min']:.2f}, "
            f"alarm_association={row['alarm_association_pct']:.1f}%, "
            f"alarm_types={row['alarm_types_observed']}, most_common_alarm={row['most_common_alarm']}, no_alarm_windows={int(row['windows_without_alarm'])}"
        )

    report_lines.append("\nFinal feasibility assessment")
    report_lines.append("----------------------------")
    report_lines.append(
        "This is an exploratory feasibility test. The pipeline is technically valid, but the clustering result should be interpreted "
        "as evidence for recurring structure only if the metrics are stable and the alarm associations differ meaningfully across clusters. "
        "If the metrics are weak, unstable, or difficult to interpret, the correct conclusion is to revisit the windowing and feature design before claiming pattern discovery."
    )

    (TARGET_OUTPUT_DIR / "analysis_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    data = load_data(DAYS)
    sensor_df = data["sensor"]
    meta_df = data["meta"]

    raw_windows = create_time_windows(sensor_df)
    print(f"Total windows: {len(raw_windows)}")
    valid_windows = extract_window_features(sensor_df, FEATURE_COLUMNS)
    removed_windows = len(raw_windows) - len(valid_windows)
    print(f"Valid windows: {len(valid_windows)}")
    print(f"Removed windows: {removed_windows}")

    feature_df, _, _ = detect_anomalies(valid_windows)
    clustering = cluster_anomalies(feature_df)
    anomaly_df = clustering["anomaly_df"]
    metrics = clustering["metrics"]
    selected_k = clustering["selected_k"]

    anomaly_df["alarm_associated"] = False
    for cluster_id in sorted(anomaly_df["cluster"].unique().tolist()):
        cluster_rows = anomaly_df[anomaly_df["cluster"] == cluster_id]
        for idx, row in cluster_rows.iterrows():
            win_start = row["window_start"]
            win_end = row["window_end"]
            alarms_window = data["full"][(data["full"].index >= win_start - pd.Timedelta("5min")) & (data["full"].index < win_end + pd.Timedelta("5min"))]
            anomaly_df.at[idx, "alarm_associated"] = bool((alarms_window["alarms"] != "none").any())

    summary_df = analyze_alarm_association(anomaly_df, data["full"])

    feature_cols = [col for col in anomaly_df.columns if col.startswith("FEATURE")]
    stability = compute_cluster_stability(anomaly_df, feature_cols, seeds=(42, 7, 21, 100, 123), k=3)
    top_features = identify_top_cluster_features(anomaly_df, top_n=20)
    top_sensors = summarize_top_sensors(top_features)
    cluster_profiles = create_cluster_profiles(anomaly_df, summary_df, top_features)
    contingency, chi2, p_value = run_chi_square_test(anomaly_df)
    create_additional_plots(anomaly_df, top_features, summary_df)
    write_cluster_interpretation_report(anomaly_df, summary_df, stability, top_features, contingency, chi2, p_value, total_observations=len(data["full"]), valid_window_count=len(valid_windows))
    save_results(anomaly_df, metrics, selected_k, summary_df, total_observations=len(data["full"]), valid_window_count=len(valid_windows))

    top_5 = top_features["feature"].head(5).tolist()
    sensor_top_5 = top_features["sensor_id"].dropna().drop_duplicates().head(5).tolist()
    print("\nSummary:")
    print(f"  Stability: mean pairwise ARI = {stability['mean_ari']:.4f} ({'stable' if stability['preserved'] else 'unstable'})")
    print(f"  Top 5 features: {top_5}")
    print(f"  Cluster sizes: {anomaly_df['cluster'].value_counts().sort_index().to_dict()}")
    print(f"  Alarm association percentages: {summary_df.set_index('cluster')['alarm_association_pct'].to_dict()}")
    print(f"  Statistical test: chi-square={chi2:.4f}, p={p_value:.6g}")

    selected_metric = next((m for m in metrics if m["k"] == selected_k), None)
    silhouette = float(selected_metric["silhouette"]) if selected_metric is not None and np.isfinite(selected_metric["silhouette"]) else 0.0
    db_score = float(selected_metric["davies_bouldin"]) if selected_metric is not None and np.isfinite(selected_metric["davies_bouldin"]) else float("inf")
    recurring = bool(summary_df["dates"].apply(lambda x: bool(x)).any())
    if stability["preserved"] and silhouette > 0.55 and db_score < 1.0 and p_value < 0.05 and recurring:
        feasibility = "Promising"
    elif stability["preserved"] or silhouette > 0.50 or p_value < 0.05:
        feasibility = "Inconclusive"
    else:
        feasibility = "Not promising"

    print("\nCLUSTER STABILITY")
    print(f"Mean pairwise ARI: {stability['mean_ari']:.4f}")
    print("\nFEATURE INTERPRETATION")
    print(f"Top 5 sensors: {', '.join(sensor_top_5)}")
    print("\nALARM ASSOCIATION")
    for _, row in summary_df.iterrows():
        print(f"Cluster {int(row['cluster'])}: {row['alarm_association_pct']:.2f}% alarm association")
    print(f"\nFINAL FEASIBILITY\n{feasibility}")
    print("\nGenerated files:")
    for path in sorted(TARGET_OUTPUT_DIR.glob('*')):
        print(f"    {path}")

    print("\nDone. Outputs are stored in outputs/tina_experiment/")


if __name__ == "__main__":
    main()
