"""Reusable data preparation and clustering functions for the dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]
RANDOM_STATE = 42


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Validate, median-fill, log-transform and standardize the six spend columns."""
    missing_columns = [col for col in FEATURES if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required spending columns: "
            + ", ".join(missing_columns)
        )

    X = df[FEATURES].copy()
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    missing_count = int(X.isna().sum().sum())
    if missing_count:
        medians = X.median(numeric_only=True)
        X = X.fillna(medians)

    if X.isna().any().any():
        raise ValueError("A spending column has no usable numeric values.")
    if (X < 0).any().any():
        raise ValueError("Spending values must be zero or greater for log transformation.")
    if len(X) < 10:
        raise ValueError("Please provide at least 10 customer records.")

    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)
    return X, X_scaled, missing_count


def evaluate_kmeans(X_scaled: np.ndarray, max_k: int = 10) -> pd.DataFrame:
    """Evaluate K values from 2 through max_k using inertia and silhouette score."""
    if len(X_scaled) < 3:
        raise ValueError("At least 3 records are required to evaluate K-Means.")

    upper_k = min(max(2, int(max_k)), len(X_scaled) - 1)
    rows: list[dict[str, float | int]] = []
    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                "K": k,
                "Inertia": float(model.inertia_),
                "Silhouette Score": float(silhouette_score(X_scaled, labels)),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(X_scaled: np.ndarray, k: int) -> tuple[np.ndarray, float, KMeans]:
    """Fit K-Means for the selected K and return labels, silhouette and model."""
    if not 2 <= int(k) < len(X_scaled):
        raise ValueError("K must be at least 2 and less than the number of records.")
    model = KMeans(n_clusters=int(k), random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = float(silhouette_score(X_scaled, labels))
    return labels, score, model


def fit_dbscan(
    X_scaled: np.ndarray, eps: float, min_samples: int
) -> tuple[np.ndarray, np.ndarray, float | None, int, int]:
    """Fit DBSCAN and return labels, sorted k-distances, optional silhouette, clusters, noise."""
    if len(X_scaled) < 2:
        raise ValueError("At least two records are required for DBSCAN.")

    effective_min_samples = min(max(2, int(min_samples)), len(X_scaled))
    neighbors = NearestNeighbors(n_neighbors=effective_min_samples)
    neighbors.fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    model = DBSCAN(eps=float(eps), min_samples=effective_min_samples)
    labels = model.fit_predict(X_scaled)
    cluster_ids = set(labels) - {-1}
    cluster_count = len(cluster_ids)
    noise_count = int(np.sum(labels == -1))

    score: float | None = None
    valid = labels != -1
    if int(valid.sum()) > 1:
        valid_labels = labels[valid]
        if 2 <= len(np.unique(valid_labels)) < int(valid.sum()):
            score = float(silhouette_score(X_scaled[valid], valid_labels))

    return labels, k_distances, score, cluster_count, noise_count


def pca_projection(X_scaled: np.ndarray) -> tuple[np.ndarray, float]:
    """Project the six-dimensional data to two dimensions for display only."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(X_scaled)
    explained = float(pca.explained_variance_ratio_.sum() * 100)
    return coordinates, explained


def make_profile(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Create average-spending profiles by cluster."""
    profile = X.copy()
    profile.insert(0, "Cluster", labels)
    return profile.groupby("Cluster").mean().round(2)


def make_download(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")
