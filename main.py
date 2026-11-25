"""Customer segmentation with K-Means and DBSCAN.

Dataset: UCI Machine Learning Repository - Wholesale Customers (ID 292)
The dataset is downloaded automatically on the first run and cached in data/.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Project settings - you can change these values after studying the results.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = PROJECT_DIR / "plots"

FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]

K_VALUES = range(2, 9)
DBSCAN_EPS = 1.2       # Starting value; review plots/dbscan_k_distance.png
DBSCAN_MIN_SAMPLES = 7
RANDOM_STATE = 42


def create_project_folders() -> None:
    """Create output folders if they do not already exist."""
    for folder in (DATA_DIR, RESULTS_DIR, PLOTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Load a local copy, or download the official UCI dataset and cache it."""
    local_candidates = [
        DATA_DIR / "Wholesale customers data.csv",
        DATA_DIR / "wholesale_customers.csv",
    ]

    for csv_path in local_candidates:
        if csv_path.exists():
            print(f"Loading local dataset: {csv_path.name}")
            return pd.read_csv(csv_path)

    print("No local dataset found. Downloading the UCI Wholesale Customers dataset...")
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "The 'ucimlrepo' package is not installed. Run:\n"
            "    python -m pip install -r requirements.txt"
        ) from exc

    try:
        wholesale = fetch_ucirepo(id=292)
        feature_data = wholesale.data.features.copy()
        target_data = wholesale.data.targets.copy()
        df = pd.concat([feature_data, target_data], axis=1)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset from UCI. Check your internet connection "
            "and try again. You can also download the CSV manually from:\n"
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/"
            "Wholesale%20customers%20data.csv\n"
            f"Original error: {exc}"
        ) from exc

    cached_path = DATA_DIR / "Wholesale customers data.csv"
    df.to_csv(cached_path, index=False)
    print(f"Dataset downloaded and saved to: {cached_path}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate, log-transform, and standardize the six spending features."""
    missing_columns = [column for column in FEATURES if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing_columns)
        )

    X = df[FEATURES].copy()
    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    missing_count = int(X.isna().sum().sum())
    if missing_count:
        warnings.warn(
            f"Found {missing_count} missing feature value(s); replacing them with "
            "the median of their respective columns.",
            stacklevel=2,
        )
        X = X.fillna(X.median(numeric_only=True))

    if (X < 0).any().any():
        raise ValueError("Spending features must be non-negative for log1p transformation.")

    # The spending data are highly skewed, so log1p reduces the influence of
    # very large values. StandardScaler then gives all six features a common scale.
    X_log = np.log1p(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    return X, X_scaled


def run_kmeans(X_scaled: np.ndarray) -> tuple[np.ndarray, int, pd.DataFrame, float]:
    """Evaluate K values, select the highest silhouette score, and fit K-Means."""
    available_k = [k for k in K_VALUES if k < len(X_scaled)]
    if not available_k:
        raise ValueError("At least three customer records are required for K-Means.")

    inertias: list[float] = []
    scores: list[float] = []

    print("\nTesting K-Means cluster counts...")
    for k in available_k:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)
        inertias.append(float(model.inertia_))
        scores.append(float(silhouette_score(X_scaled, labels)))
        print(f"  K={k}: inertia={model.inertia_:.2f}, silhouette={scores[-1]:.4f}")

    best_k_index = int(np.argmax(scores))
    best_k = available_k[best_k_index]

    evaluation = pd.DataFrame(
        {
            "K": available_k,
            "Inertia": inertias,
            "Silhouette_Score": scores,
        }
    )

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = final_model.fit_predict(X_scaled)
    best_score = float(scores[best_k_index])

    # Save Elbow and Silhouette evaluation charts.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(available_k, inertias, marker="o")
    axes[0].set_title("K-Means - Elbow Method")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(alpha=0.25)

    axes[1].plot(available_k, scores, marker="o")
    axes[1].set_title("K-Means - Silhouette Scores")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "kmeans_model_selection.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return labels, best_k, evaluation, best_score


def run_dbscan(X_scaled: np.ndarray) -> tuple[np.ndarray, float | None, int, int]:
    """Create a k-distance chart and fit DBSCAN using the configured parameters."""
    if len(X_scaled) <= DBSCAN_MIN_SAMPLES:
        raise ValueError("There are not enough rows for the selected DBSCAN min_samples value.")

    # The point itself counts as one of its nearest neighbors in this calculation.
    neighbors = NearestNeighbors(n_neighbors=DBSCAN_MIN_SAMPLES)
    neighbors.fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, len(k_distances) + 1), k_distances)
    ax.axhline(DBSCAN_EPS, linestyle="--", label=f"Current eps = {DBSCAN_EPS}")
    ax.set_title(f"DBSCAN - {DBSCAN_MIN_SAMPLES}-Nearest-Neighbor Distance")
    ax.set_xlabel("Customers (sorted by distance)")
    ax.set_ylabel(f"Distance to {DBSCAN_MIN_SAMPLES}th nearest neighbor")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "dbscan_k_distance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = model.fit_predict(X_scaled)

    cluster_labels = set(labels) - {-1}
    n_clusters = len(cluster_labels)
    n_noise = int(np.sum(labels == -1))

    # Silhouette is calculated only for non-noise records, and only when at
    # least two clusters remain. This score is not directly comparable with
    # K-Means because DBSCAN noise records are excluded.
    valid = labels != -1
    valid_labels = labels[valid]
    valid_cluster_count = len(set(valid_labels))
    dbscan_score: float | None = None
    if valid.sum() > 1 and 2 <= valid_cluster_count < valid.sum():
        dbscan_score = float(silhouette_score(X_scaled[valid], valid_labels))

    return labels, dbscan_score, n_clusters, n_noise


def create_visualization(
    X_scaled: np.ndarray,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
) -> None:
    """Project six-dimensional data to two dimensions for plotting only."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    explained = float(pca.explained_variance_ratio_.sum() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    first = axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap="viridis", s=35, alpha=0.8
    )
    axes[0].set_title("K-Means Customer Clusters")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    fig.colorbar(first, ax=axes[0], label="Cluster")

    normal = dbscan_labels != -1
    noise = dbscan_labels == -1
    if np.any(normal):
        second = axes[1].scatter(
            X_2d[normal, 0],
            X_2d[normal, 1],
            c=dbscan_labels[normal],
            cmap="viridis",
            s=35,
            alpha=0.8,
        )
        fig.colorbar(second, ax=axes[1], label="Cluster")
    if np.any(noise):
        axes[1].scatter(
            X_2d[noise, 0],
            X_2d[noise, 1],
            c="black",
            marker="x",
            s=45,
            label="Noise (-1)",
        )
        axes[1].legend()

    axes[1].set_title("DBSCAN Customer Clusters")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")

    fig.suptitle(f"PCA visualization - {explained:.1f}% variance represented")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_comparison_pca.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_results(
    df: pd.DataFrame,
    X: pd.DataFrame,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    kmeans_evaluation: pd.DataFrame,
    kmeans_score: float,
    best_k: int,
    dbscan_score: float | None,
    dbscan_clusters: int,
    dbscan_noise: int,
) -> None:
    """Save customer labels, cluster profiles, evaluation metrics, and summary."""
    results = df.copy()
    results["KMeans_Cluster"] = kmeans_labels
    results["DBSCAN_Cluster"] = dbscan_labels
    results.to_csv(RESULTS_DIR / "customer_clustering_results.csv", index=False)

    kmeans_profile = X.assign(KMeans_Cluster=kmeans_labels).groupby("KMeans_Cluster").mean()
    kmeans_profile.to_csv(RESULTS_DIR / "kmeans_cluster_profiles.csv")

    dbscan_profile = X.assign(DBSCAN_Cluster=dbscan_labels).groupby("DBSCAN_Cluster").mean()
    dbscan_profile.to_csv(RESULTS_DIR / "dbscan_cluster_profiles.csv")

    kmeans_evaluation.to_csv(RESULTS_DIR / "kmeans_evaluation.csv", index=False)

    metrics = pd.DataFrame(
        [
            {
                "Algorithm": "K-Means",
                "Number_of_Clusters": best_k,
                "Noise_Points": 0,
                "Silhouette_Score": kmeans_score,
                "Evaluation_Note": "Score calculated using all customers",
            },
            {
                "Algorithm": "DBSCAN",
                "Number_of_Clusters": dbscan_clusters,
                "Noise_Points": dbscan_noise,
                "Silhouette_Score": dbscan_score,
                "Evaluation_Note": (
                    "DBSCAN score excludes noise points; compare cautiously"
                    if dbscan_score is not None
                    else "Score unavailable: DBSCAN found fewer than two non-noise clusters"
                ),
            },
        ]
    )
    metrics.to_csv(RESULTS_DIR / "algorithm_comparison.csv", index=False)

    summary_lines = [
        "CUSTOMER CLUSTERING - RUN SUMMARY",
        "=================================",
        f"Customers analysed: {len(df)}",
        f"Features used: {', '.join(FEATURES)}",
        "Preprocessing: log1p transformation followed by StandardScaler",
        "Visualization: PCA projection to two dimensions (clustering used all six features)",
        "",
        "K-MEANS",
        f"Selected K: {best_k}",
        f"Silhouette score: {kmeans_score:.4f}",
        "",
        "DBSCAN",
        f"Epsilon: {DBSCAN_EPS}",
        f"Min samples: {DBSCAN_MIN_SAMPLES}",
        f"Clusters found: {dbscan_clusters}",
        f"Noise points: {dbscan_noise}",
        f"Silhouette score (excluding noise): {dbscan_score if dbscan_score is not None else 'Not available'}",
        "",
        "See the results/ and plots/ folders for CSV files and charts.",
    ]
    (RESULTS_DIR / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> int:
    print("=" * 68)
    print("CUSTOMER SEGMENTATION: K-MEANS AND DBSCAN")
    print("=" * 68)
    create_project_folders()

    try:
        df = load_dataset()
        print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("\nFirst five records:")
        print(df.head().to_string(index=False))

        X, X_scaled = prepare_features(df)
        print(f"\nFeatures selected: {len(FEATURES)}")
        print(f"Missing values after preparation: {int(X.isna().sum().sum())}")
        print("Log transformation and standardization completed.")

        kmeans_labels, best_k, kmeans_eval, kmeans_score = run_kmeans(X_scaled)
        print(f"\nK-Means selected K = {best_k}")
        print(f"K-Means silhouette score = {kmeans_score:.4f}")
        print("K-Means customer counts:")
        print(pd.Series(kmeans_labels).value_counts().sort_index().to_string())

        dbscan_labels, dbscan_score, dbscan_clusters, dbscan_noise = run_dbscan(X_scaled)
        print("\nDBSCAN results:")
        print(f"  eps = {DBSCAN_EPS}")
        print(f"  min_samples = {DBSCAN_MIN_SAMPLES}")
        print(f"  clusters found = {dbscan_clusters}")
        print(f"  noise points = {dbscan_noise}")
        if dbscan_score is None:
            print("  silhouette score = unavailable (need at least two non-noise clusters)")
        else:
            print(f"  silhouette score excluding noise = {dbscan_score:.4f}")
        print("DBSCAN customer counts (-1 means noise):")
        print(pd.Series(dbscan_labels).value_counts().sort_index().to_string())

        create_visualization(X_scaled, kmeans_labels, dbscan_labels)
        save_results(
            df=df,
            X=X,
            kmeans_labels=kmeans_labels,
            dbscan_labels=dbscan_labels,
            kmeans_evaluation=kmeans_eval,
            kmeans_score=kmeans_score,
            best_k=best_k,
            dbscan_score=dbscan_score,
            dbscan_clusters=dbscan_clusters,
            dbscan_noise=dbscan_noise,
        )

        print("\n" + "=" * 68)
        print("PROJECT COMPLETED SUCCESSFULLY")
        print("=" * 68)
        print(f"Results folder: {RESULTS_DIR}")
        print(f"Plots folder:   {PLOTS_DIR}")
        print("\nImportant: open plots/dbscan_k_distance.png and inspect the bend in the curve.")
        print("If needed, change DBSCAN_EPS near the top of main.py and run again.")
        return 0

    except Exception as exc:
        print("\nPROJECT COULD NOT COMPLETE")
        print(f"Reason: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
