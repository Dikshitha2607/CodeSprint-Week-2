import os
import tarfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path(__file__).resolve().parent
ARCHIVE_PATH = DATA_ROOT / "pill.tar.xz"
OUTPUT_DIR = DATA_ROOT / "outputs" / "pill_anomaly_experiment"
CACHE_DIR = DATA_ROOT / "cache"
EMBEDDING_PATH = CACHE_DIR / "pill_anomaly_embeddings.npz"
DEFECT_REGION_EMBEDDING_PATH = CACHE_DIR / "pill_defect_region_embeddings.npz"
DEFECT_REGION_OUTPUT_DIR = OUTPUT_DIR / "defect_region_experiment"
REPORT_PATH = OUTPUT_DIR / "analysis_report.txt"
PADDING_RATIO = 0.15


def locate_mvtec_dataset(root: Path) -> Path:
    """Find the extracted pill dataset folder using the actual on-disk structure."""
    if (root / "pill").exists() and (root / "pill" / "test").exists():
        return root / "pill"

    for candidate in sorted(root.rglob("pill")):
        if (candidate / "test").exists() and (candidate / "train").exists():
            return candidate

    raise FileNotFoundError(f"Could not find a MVTec 'pill' dataset under {root}")


def extract_archive_if_needed(archive_path: Path, extract_root: Path) -> Path:
    """Extract the downloaded MVTec pill archive to a sibling folder if needed."""
    if archive_path.exists() and not extract_root.exists():
        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(path=extract_root.parent)
    return locate_mvtec_dataset(extract_root.parent)


def collect_image_files(directory: Path):
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in image_extensions],
        key=lambda p: p.name,
    )


def find_mask_for_image(image_path: Path, ground_truth_root: Path | None):
    if ground_truth_root is None:
        return None

    label_folder = ground_truth_root / image_path.parent.name
    if not label_folder.exists():
        return None

    stem = image_path.stem
    candidate_stems = {stem, f"{stem}_mask", f"{stem}_gt"}
    candidates = []
    for p in label_folder.iterdir():
        if not p.is_file():
            continue
        if p.stem in candidate_stems:
            candidates.append(p)

    if not candidates:
        # Some MVTec masks are named with the image stem plus a suffix, e.g. 000_mask.png.
        for p in label_folder.iterdir():
            if not p.is_file():
                continue
            if p.stem.startswith(f"{stem}_") or p.stem.endswith("_mask"):
                candidates.append(p)

    if candidates:
        return sorted(candidates, key=lambda p: p.name)[0]
    return None


def load_anomaly_mask(image_path: Path, ground_truth_root: Path | None):
    """Load and validate the anomaly mask for an anomalous image."""
    if ground_truth_root is None:
        return None, "ground_truth directory not found"

    mask_path = find_mask_for_image(image_path, ground_truth_root)
    if mask_path is None:
        return None, f"no mask for {image_path.name}"

    try:
        mask = np.array(Image.open(mask_path).convert("L"))
    except Exception as exc:
        return None, f"could not read mask: {exc}"

    if mask.size == 0:
        return None, "empty mask"

    if np.count_nonzero(mask) == 0:
        return None, "mask contains no anomaly pixels"

    return mask, "ok"


def crop_defect_region(image_path: Path, mask: np.ndarray, padding_ratio: float = PADDING_RATIO):
    """Crop the bounding box of the anomaly region with configurable padding."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"No non-zero mask pixels found for {image_path}")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    pad_x = max(1, int((x_max - x_min + 1) * padding_ratio))
    pad_y = max(1, int((y_max - y_min + 1) * padding_ratio))

    left = max(0, x_min - pad_x)
    right = min(img_w, x_max + pad_x + 1)
    top = max(0, y_min - pad_y)
    bottom = min(img_h, y_max + pad_y + 1)

    crop = img.crop((left, top, right, bottom))
    return crop


def load_pill_dataset(dataset_root: Path) -> pd.DataFrame:
    """Load MVTec pill test images into a DataFrame, separating good vs anomalous images."""
    test_root = dataset_root / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"No test folder found under {dataset_root}")

    ground_truth_root = dataset_root / "ground_truth" if (dataset_root / "ground_truth").exists() else None

    rows = []
    normal_dir = test_root / "good"
    if normal_dir.exists():
        for img_path in collect_image_files(normal_dir):
            rows.append(
                {
                    "image_path": str(img_path),
                    "image": img_path,
                    "defect_label": "good",
                    "original_category": "good",
                    "is_anomalous": False,
                    "mask_path": None,
                    "source_folder": "test/good",
                }
            )

    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue

        for img_path in collect_image_files(defect_dir):
            mask_path = find_mask_for_image(img_path, ground_truth_root)
            rows.append(
                {
                    "image_path": str(img_path),
                    "image": img_path,
                    "defect_label": defect_dir.name,
                    "original_category": defect_dir.name,
                    "is_anomalous": True,
                    "mask_path": str(mask_path) if mask_path else None,
                    "source_folder": defect_dir.name,
                }
            )

    df = pd.DataFrame(rows)
    df["image_name"] = df["image"].apply(lambda p: Path(p).name)
    return df


def summarize_dataset(df: pd.DataFrame):
    total = len(df)
    normal = int((df["is_anomalous"] == False).sum())
    anomalous = int((df["is_anomalous"] == True).sum())
    defect_counts = df[df["is_anomalous"]].groupby("defect_label").size().sort_values(ascending=False)
    summary = {
        "total_test_images": total,
        "num_normal_images": normal,
        "num_anomalous_images": anomalous,
        "defect_counts": defect_counts,
    }
    return summary


def display_example_images(df: pd.DataFrame, output_dir: Path, max_per_group: int = 3):
    anomalous_groups = df[df["is_anomalous"]].groupby("defect_label")
    output_dir.mkdir(parents=True, exist_ok=True)

    for defect_name, group in anomalous_groups:
        sample_paths = group["image"].head(max_per_group).tolist()
        fig, axes = plt.subplots(1, len(sample_paths), figsize=(4 * len(sample_paths), 4))
        if len(sample_paths) == 1:
            axes = [axes]
        for ax, image_path in zip(axes, sample_paths):
            img = Image.open(image_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(defect_name)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"example_{defect_name}.png", dpi=200)
        plt.close(fig)


def get_preprocess_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_embedding_model():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model


def load_or_extract_embeddings(anomalous_df: pd.DataFrame, embedding_path: Path):
    if embedding_path.exists():
        try:
            with np.load(embedding_path, allow_pickle=False) as cached:
                if "embeddings" in cached and "paths" in cached:
                    embeddings = cached["embeddings"]
                    paths = cached["paths"]
                    if embeddings.ndim == 2 and embeddings.shape[0] == len(anomalous_df):
                        return embeddings, [str(p) for p in paths.tolist()]
        except (ValueError, OSError, KeyError):
            pass

        try:
            embedding_path.unlink()
        except FileNotFoundError:
            pass

    model = build_embedding_model()
    transform = get_preprocess_transform()
    batch_size = 32
    features = []
    image_paths = []

    with torch.no_grad():
        for i in range(0, len(anomalous_df), batch_size):
            batch = anomalous_df.iloc[i : i + batch_size]
            tensors = []
            for image_path in batch["image"]:
                img = Image.open(image_path).convert("RGB")
                tensors.append(transform(img))
            x = torch.stack(tensors).to(DEVICE)
            embeddings = model(x)
            features.append(embeddings.cpu().numpy())
            image_paths.extend(batch["image_path"].tolist())

    features_array = np.concatenate(features, axis=0)
    os.makedirs(embedding_path.parent, exist_ok=True)
    np.savez(embedding_path, embeddings=features_array, paths=np.array(image_paths, dtype=str))
    return features_array, image_paths


def evaluate_kmeans(embeddings: np.ndarray, k_values, max_k=None):
    if max_k is not None:
        k_values = [k for k in k_values if k <= max_k]

    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    results = []
    for k in k_values:
        if k >= len(X):
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
        db = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
        results.append({"k": k, "silhouette": sil, "davies_bouldin": db, "labels": labels})
    return results, X


def run_dbscan(embeddings: np.ndarray):
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    for eps in [0.5, 1.0, 1.5, 2.0]:
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(X)
        unique = np.unique(labels)
        if len(unique[unique != -1]) >= 2:
            sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 and len(set(labels) - {-1}) > 1 else float("nan")
            db = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 and len(set(labels) - {-1}) > 1 else float("nan")
            return {"eps": eps, "labels": labels, "silhouette": sil, "davies_bouldin": db, "n_clusters": len(set(labels) - {-1})}
    return {"eps": None, "labels": None, "silhouette": float("nan"), "davies_bouldin": float("nan"), "n_clusters": 1}


def plot_cluster_scatter(embeddings: np.ndarray, labels: np.ndarray, title: str, output_path: Path):
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=45, alpha=0.8)
    plt.colorbar(label="Cluster ID")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def find_representative_images(embeddings: np.ndarray, labels: np.ndarray, image_paths: list[str], defect_labels: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_info = {}

    for cluster_id in sorted(np.unique(labels)):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue

        centroid = embeddings[indices].mean(axis=0)
        distances = np.linalg.norm(embeddings[indices] - centroid, axis=1)
        representative_idx = indices[np.argsort(distances)[:5]]

        fig, axes = plt.subplots(1, min(5, len(representative_idx)), figsize=(18, 4))
        if len(representative_idx) == 1:
            axes = [axes]

        for ax, idx in zip(axes, representative_idx):
            image_path = image_paths[idx]
            img = Image.open(image_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"{defect_labels[idx]}\n{Path(image_path).name}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Cluster {cluster_id} (n={len(indices)})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(output_dir / f"cluster_{cluster_id}_representatives.png", dpi=200)
        plt.close(fig)

        cluster_info[cluster_id] = {
            "size": len(indices),
            "representatives": [image_paths[i] for i in representative_idx],
            "defect_labels": [defect_labels[i] for i in representative_idx],
        }
    return cluster_info


def evaluate_cluster_label_relationship(labels: np.ndarray, defect_labels: list[str]):
    contingency = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(defect_labels, name="defect_label"))
    ari = adjusted_rand_score(defect_labels, labels)
    nmi = normalized_mutual_info_score(defect_labels, labels)
    return contingency, ari, nmi


def generate_report(summary: dict, embedding_model_name: str, embedding_dim: int, clustering_results: dict, candidate_k: int, candidate_labels: np.ndarray, contour_df: pd.DataFrame, ari: float, nmi: float, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append("Industrial Anomaly Pattern Discovery - MVTec Pill Prototype")
    report_lines.append("=" * 80)
    report_lines.append(f"Number of normal images: {summary['num_normal_images']}")
    report_lines.append(f"Number of anomalous images: {summary['num_anomalous_images']}")
    report_lines.append(f"Original defect categories: {list(summary['defect_counts'].index)}")
    report_lines.append(f"Embedding model: {embedding_model_name}")
    report_lines.append(f"Embedding dimensionality: {embedding_dim}")
    report_lines.append(f"Clustering algorithms tested: K-Means, DBSCAN")
    report_lines.append(f"K values tested: {list(clustering_results['k_values'])}")
    report_lines.append(f"Silhouette/Davies-Bouldin summary: {clustering_results['metric_summary']}")
    report_lines.append(f"Final candidate clustering: K-Means with k={candidate_k}")
    report_lines.append(f"Cluster sizes: {dict(zip(sorted(np.unique(candidate_labels)), np.bincount(candidate_labels)))}")
    report_lines.append(f"ARI vs original labels: {ari:.4f}")
    report_lines.append(f"NMI vs original labels: {nmi:.4f}")
    report_lines.append("")
    report_lines.append("Contingency table (cluster x original defect label):")
    report_lines.append(contour_df.to_string())
    report_lines.append("")
    report_lines.append("Is the Core Idea Valid?")
    report_lines.append("-" * 40)

    # Evidence-based verdict from clustering metrics and label comparisons.
    if ari > 0.15 and nmi > 0.2:
        verdict = "Promising — continue with this approach"
    elif ari > 0.0 or nmi > 0.0:
        verdict = "Inconclusive — needs further testing"
    else:
        verdict = "Weak evidence — consider switching dataset"

    report_lines.append(f"Likely verdict: {verdict}")
    report_lines.append(
        "The clustering is only considered meaningful if the representative images inside each cluster look visually coherent and distinct from other clusters. "
        "A high label agreement score alone is not enough to claim the unsupervised clusters are semantically meaningful."
    )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    return verdict


def extract_defect_region_embeddings(anomalous_df: pd.DataFrame, dataset_root: Path, embedding_path: Path, padding_ratio: float = PADDING_RATIO):
    """Extract MobileNetV2 embeddings from masked defect crops and cache them separately."""
    if embedding_path.exists():
        try:
            with np.load(embedding_path, allow_pickle=False) as cached:
                if "embeddings" in cached and "paths" in cached and "defect_labels" in cached:
                    embeddings = cached["embeddings"]
                    paths = cached["paths"]
                    defect_labels = cached["defect_labels"]
                    if embeddings.ndim == 2 and embeddings.shape[0] == len(anomalous_df):
                        return embeddings, [str(p) for p in paths.tolist()], [str(d) for d in defect_labels.tolist()]
        except (ValueError, OSError, KeyError):
            pass

        try:
            embedding_path.unlink()
        except FileNotFoundError:
            pass

    ground_truth_root = dataset_root / "ground_truth" if (dataset_root / "ground_truth").exists() else None
    model = build_embedding_model()
    transform = get_preprocess_transform()
    batch_size = 32
    features = []
    image_paths = []
    defect_labels = []
    skipped = []

    with torch.no_grad():
        for _, row in anomalous_df.iterrows():
            image_path = Path(row["image_path"])
            mask, reason = load_anomaly_mask(image_path, ground_truth_root)
            if mask is None:
                skipped.append({"filename": image_path.name, "reason": reason})
                continue

            try:
                crop = crop_defect_region(image_path, mask, padding_ratio=padding_ratio)
            except Exception as exc:
                skipped.append({"filename": image_path.name, "reason": f"crop failed: {exc}"})
                continue

            tensor = transform(crop).unsqueeze(0).to(DEVICE)
            embedding = model(tensor).cpu().numpy()[0]
            features.append(embedding)
            image_paths.append(str(image_path))
            defect_labels.append(str(row["defect_label"]))

    if len(features) == 0:
        raise ValueError("No defect-region embeddings were successfully extracted from the anomalous images.")

    features_array = np.vstack(features)
    os.makedirs(embedding_path.parent, exist_ok=True)
    np.savez(
        embedding_path,
        embeddings=features_array,
        paths=np.array(image_paths, dtype=str),
        defect_labels=np.array(defect_labels, dtype=str),
    )
    return features_array, image_paths, defect_labels, skipped


def evaluate_defect_region_clustering(embeddings: np.ndarray, k_values, max_k=None):
    return evaluate_kmeans(embeddings, k_values, max_k=max_k)


def create_defect_region_contact_sheets(cluster_labels: np.ndarray, image_paths: list[str], defect_labels: list[str], embeddings: np.ndarray, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_to_paths = {}
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_to_paths.setdefault(int(cluster_id), []).append((idx, image_paths[idx], defect_labels[idx]))

    for cluster_id in sorted(cluster_to_paths):
        members = cluster_to_paths[cluster_id]
        centroid = embeddings[[idx for idx, _, _ in members]].mean(axis=0)
        distances = np.linalg.norm(embeddings[[idx for idx, _, _ in members]] - centroid, axis=1)
        ranking = sorted(range(len(members)), key=lambda i: distances[i])[:5]

        representative_members = [members[i] for i in ranking]
        fig, axes = plt.subplots(1, min(5, len(representative_members)), figsize=(18, 4))
        if len(representative_members) == 1:
            axes = [axes]

        for ax, (_, path, defect_label) in zip(axes, representative_members):
            crop_path = Path(path).with_name(Path(path).name)
            actual_crop = Image.open(path).convert("RGB")
            ax.imshow(actual_crop)
            ax.set_title(f"{defect_label}\n{Path(path).name}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Defect-region cluster {cluster_id} (n={len(members)})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(output_dir / f"cluster_{cluster_id}_representatives.png", dpi=200)
        plt.close(fig)


def compare_experiments(whole_image_metrics: dict, defect_region_metrics: dict):
    comparison_rows = [
        {
            "Experiment": "Whole Image",
            "Best K": whole_image_metrics["best_k"],
            "Silhouette": whole_image_metrics["best_silhouette"],
            "Davies-Bouldin": whole_image_metrics["best_db"],
            "ARI": whole_image_metrics["ari"],
            "NMI": whole_image_metrics["nmi"],
        },
        {
            "Experiment": "Defect Region",
            "Best K": defect_region_metrics["best_k"],
            "Silhouette": defect_region_metrics["best_silhouette"],
            "Davies-Bouldin": defect_region_metrics["best_db"],
            "ARI": defect_region_metrics["ari"],
            "NMI": defect_region_metrics["nmi"],
        },
    ]
    return pd.DataFrame(comparison_rows)


def generate_defect_region_report(summary: dict, whole_image_metrics: dict, defect_region_metrics: dict, contingency: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = compare_experiments(whole_image_metrics, defect_region_metrics)

    report_lines = []
    report_lines.append("Industrial Anomaly Pattern Discovery - Defect Region Experiment")
    report_lines.append("=" * 90)
    report_lines.append(f"Number of normal images: {summary['num_normal_images']}")
    report_lines.append(f"Number of anomalous images: {summary['num_anomalous_images']}")
    report_lines.append(f"Original defect categories: {list(summary['defect_counts'].index)}")
    report_lines.append(f"Embedding model: MobileNetV2 (ImageNet pretraining, final classification layer removed)")
    report_lines.append(f"Embedding dimensionality: {defect_region_metrics['embedding_dim']}")
    report_lines.append(f"Padding ratio: {PADDING_RATIO}")
    report_lines.append(f"Successfully processed defect-region images: {defect_region_metrics['success_count']}")
    report_lines.append(f"Skipped defect-region images: {defect_region_metrics['skipped_count']}")
    report_lines.append(f"K values tested: {defect_region_metrics['k_values']}")
    report_lines.append(f"K-Means metrics: {defect_region_metrics['k_metrics']}")
    report_lines.append(f"Best candidate K: {defect_region_metrics['best_k']} (silhouette={defect_region_metrics['best_silhouette']:.4f}, DB={defect_region_metrics['best_db']:.4f})")
    report_lines.append(f"ARI vs original labels: {defect_region_metrics['ari']:.4f}")
    report_lines.append(f"NMI vs original labels: {defect_region_metrics['nmi']:.4f}")
    report_lines.append("")
    report_lines.append("Contingency table (cluster x original defect label):")
    report_lines.append(contingency.to_string())
    report_lines.append("")
    report_lines.append("WHOLE IMAGE VS DEFECT REGION")
    report_lines.append("-" * 90)
    report_lines.append(comparison_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("1. Did defect-region embeddings improve cluster separation? " + str(defect_region_metrics['best_silhouette'] > whole_image_metrics['best_silhouette']))
    report_lines.append("2. Did the clusters become more balanced? " + str(defect_region_metrics['cluster_size_balance'] >= whole_image_metrics['cluster_size_balance']))
    report_lines.append("3. Did ARI/NMI increase? " + str((defect_region_metrics['ari'] > whole_image_metrics['ari']) or (defect_region_metrics['nmi'] > whole_image_metrics['nmi'])))
    report_lines.append("4. Did the discovered clusters become less dominated by one existing defect category? " + str(defect_region_metrics['cluster_label_diversity'] > whole_image_metrics['cluster_label_diversity']))
    report_lines.append("5. Are there signs that the original whole-image embeddings captured irrelevant pill appearance or background? Review the representative defect-region crops and cluster assignments.")
    report_lines.append("6. Are the discovered patterns actually worth pursuing? Only if the representative defect-region crops are visually coherent and distinct across clusters.")
    report_lines.append("")
    report_lines.append("Methodological limitation:")
    report_lines.append("The ground-truth MVTec masks are used only as an offline diagnostic experiment to determine whether focusing on the anomaly region produces more meaningful visual representations. These masks are not available for a genuinely new uploaded image in the final application. If the defect-region experiment proves promising, the eventual application would need an actual anomaly-localization or defect-detection method before clustering.")
    report_lines.append("")

    if defect_region_metrics["ari"] >= whole_image_metrics["ari"] and defect_region_metrics["nmi"] >= whole_image_metrics["nmi"] and defect_region_metrics["best_silhouette"] > whole_image_metrics["best_silhouette"]:
        verdict = "Promising — continue with this approach"
    elif defect_region_metrics["best_silhouette"] > 0.25 or defect_region_metrics["ari"] > 0.05 or defect_region_metrics["nmi"] > 0.1:
        verdict = "Inconclusive — needs further testing"
    else:
        verdict = "Weak evidence — consider switching dataset"

    report_lines.append(f"Final verdict: {verdict}")
    (output_dir / "analysis_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    return verdict


def run_defect_region_experiment(dataset_root: Path, anomalous_df: pd.DataFrame, whole_image_metrics: dict):
    output_dir = DEFECT_REGION_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    defect_region_cache = DEFECT_REGION_EMBEDDING_PATH
    embeddings, paths, defect_labels, skipped = extract_defect_region_embeddings(anomalous_df, dataset_root, defect_region_cache, padding_ratio=PADDING_RATIO)
    successful_count = len(paths)
    skipped_count = len(skipped)

    if successful_count == 0:
        raise ValueError("No defect-region embeddings were extracted successfully.")

    k_values = list(range(2, min(8, successful_count) + 1))
    k_results, scaled = evaluate_defect_region_clustering(embeddings, k_values)
    k_metrics = pd.DataFrame([
        {"k": item["k"], "silhouette": item["silhouette"], "davies_bouldin": item["davies_bouldin"]} for item in k_results
    ])

    if k_metrics.empty:
        raise ValueError("No K-Means results were generated for the defect-region embeddings.")

    best_row = k_metrics.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    candidate_k = int(best_row["k"])
    candidate_labels = next(item["labels"] for item in k_results if item["k"] == candidate_k)

    # Visualization
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(scaled)
    plt.figure(figsize=(8, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=candidate_labels, cmap="tab10", s=45, alpha=0.8)
    plt.colorbar(label="Cluster ID")
    plt.title(f"Defect-region K-Means clusters (k={candidate_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_scatter.png", dpi=200)
    plt.close()

    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(embeddings) // 10)))
        reduced_tsne = tsne.fit_transform(scaled)
        plt.figure(figsize=(8, 7))
        plt.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], c=candidate_labels, cmap="tab10", s=45, alpha=0.8)
        plt.colorbar(label="Cluster ID")
        plt.title(f"Defect-region t-SNE clusters (k={candidate_k})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_scatter.png", dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Warning: t-SNE failed for defect-region embeddings: {exc}")

    create_defect_region_contact_sheets(candidate_labels, paths, defect_labels, embeddings, output_dir / "cluster_representatives")

    contingency, ari, nmi = evaluate_cluster_label_relationship(candidate_labels, defect_labels)

    cluster_size_balance = float(np.std(np.bincount(candidate_labels))) if len(np.unique(candidate_labels)) > 1 else 0.0
    cluster_label_diversity = float(len(np.unique(defect_labels)))

    defect_region_metrics = {
        "best_k": candidate_k,
        "best_silhouette": float(best_row["silhouette"]),
        "best_db": float(best_row["davies_bouldin"]),
        "ari": float(ari),
        "nmi": float(nmi),
        "embedding_dim": embeddings.shape[1],
        "k_values": k_values,
        "k_metrics": k_metrics.to_dict(orient="records"),
        "success_count": successful_count,
        "skipped_count": skipped_count,
        "cluster_size_balance": cluster_size_balance,
        "cluster_label_diversity": cluster_label_diversity,
    }

    summary = summarize_dataset(anomalous_df)
    verdict = generate_defect_region_report(summary, whole_image_metrics, defect_region_metrics, contingency, output_dir)

    print("\nDefect-region experiment summary:")
    print(f"- anomalous images: {summary['num_anomalous_images']}")
    print(f"- successfully cropped regions: {successful_count}")
    print(f"- skipped: {skipped_count}")
    print(f"- embedding shape: {embeddings.shape}")
    print("- K-Means results:")
    print(k_metrics.to_string(index=False))
    print(f"- ARI: {ari:.4f}")
    print(f"- NMI: {nmi:.4f}")
    print(f"- comparison with whole-image experiment: {whole_image_metrics['best_k']} vs {candidate_k} best K")
    print(f"- verdict: {verdict}")

    if skipped:
        print("- skipped files:")
        for item in skipped[:10]:
            print(f"  * {item['filename']}: {item['reason']}")

    return defect_region_metrics

def build_whole_image_metrics(anomalous_df: pd.DataFrame):
    embeddings, paths = load_or_extract_embeddings(anomalous_df, EMBEDDING_PATH)
    k_values = list(range(2, min(8, len(anomalous_df)) + 1))
    k_results, scaled = evaluate_kmeans(embeddings, k_values)
    k_metrics = pd.DataFrame([
        {"k": item["k"], "silhouette": item["silhouette"], "davies_bouldin": item["davies_bouldin"]} for item in k_results
    ])
    best_row = k_metrics.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    candidate_k = int(best_row["k"])
    candidate_labels = next(item["labels"] for item in k_results if item["k"] == candidate_k)
    contingency, ari, nmi = evaluate_cluster_label_relationship(candidate_labels, anomalous_df["defect_label"].tolist())
    cluster_size_balance = float(np.std(np.bincount(candidate_labels))) if len(np.unique(candidate_labels)) > 1 else 0.0
    cluster_label_diversity = float(len(np.unique(anomalous_df["defect_label"])))
    return {
        "best_k": candidate_k,
        "best_silhouette": float(best_row["silhouette"]),
        "best_db": float(best_row["davies_bouldin"]),
        "ari": float(ari),
        "nmi": float(nmi),
        "cluster_size_balance": cluster_size_balance,
        "cluster_label_diversity": cluster_label_diversity,
        "embedding_dim": embeddings.shape[1],
        "k_values": k_values,
        "k_metrics": k_metrics.to_dict(orient="records"),
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dataset_root = extract_archive_if_needed(ARCHIVE_PATH, DATA_ROOT / "pill")
    df = load_pill_dataset(dataset_root)
    summary = summarize_dataset(df)

    print("Dataset summary:")
    print(f"- total_test_images: {summary['total_test_images']}")
    print(f"- num_normal_images: {summary['num_normal_images']}")
    print(f"- num_anomalous_images: {summary['num_anomalous_images']}")
    print("- defect_counts:")
    print(summary['defect_counts'])

    display_example_images(df, OUTPUT_DIR / "defect_examples", max_per_group=3)

    anomalous_df = df[df["is_anomalous"]].copy().reset_index(drop=True)
    embeddings, paths = load_or_extract_embeddings(anomalous_df, EMBEDDING_PATH)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    k_values = list(range(2, min(8, len(anomalous_df)) + 1))
    k_results, scaled = evaluate_kmeans(embeddings, k_values)
    metric_rows = []
    for item in k_results:
        metric_rows.append({"k": item["k"], "silhouette": item["silhouette"], "davies_bouldin": item["davies_bouldin"]})
    k_metrics = pd.DataFrame(metric_rows)
    print("\nK-Means evaluation:")
    print(k_metrics.to_string(index=False))

    best_row = k_metrics.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    candidate_k = int(best_row["k"])
    candidate_labels = next(item["labels"] for item in k_results if item["k"] == candidate_k)

    dbscan_result = run_dbscan(embeddings)
    print("\nDBSCAN evaluation:")
    print(dbscan_result)

    plot_cluster_scatter(scaled, candidate_labels, f"Unsupervised K-Means clusters (k={candidate_k})", OUTPUT_DIR / "cluster_scatter.png")

    cluster_dir = OUTPUT_DIR / "cluster_representatives"
    cluster_info = find_representative_images(embeddings, candidate_labels, paths, anomalous_df["defect_label"].tolist(), cluster_dir)
    print("\nCluster representative images saved to:", cluster_dir)

    contingency, ari, nmi = evaluate_cluster_label_relationship(candidate_labels, anomalous_df["defect_label"].tolist())
    print("\nCluster vs original defect label contingency table:")
    print(contingency)
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")

    clustering_results = {
        "k_values": k_values,
        "metric_summary": k_metrics.to_dict(orient="records"),
    }
    verdict = generate_report(
        summary,
        "MobileNetV2 (ImageNet pretraining, final classification layer removed)",
        embeddings.shape[1],
        clustering_results,
        candidate_k,
        candidate_labels,
        contingency,
        ari,
        nmi,
        OUTPUT_DIR,
    )
    print("\nReport saved to:", REPORT_PATH)
    print("\nVerdict:", verdict)

    whole_image_metrics = build_whole_image_metrics(anomalous_df)
    print("\nRunning second experiment: defect-region embeddings")
    run_defect_region_experiment(dataset_root, anomalous_df, whole_image_metrics)

if __name__ == "__main__":
    main()
