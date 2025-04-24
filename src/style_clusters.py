"""
Playing-style archetypes: KMeans clustering of each position group on
their standardized per-90 profile — same silhouette-score-driven approach
used to pick K for the Spotify lyrics clustering project, applied here to
find "types" of forward, midfielder, and defender rather than genres.
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from similarity import FEATURE_COLS


def sweep_k(X_scaled, k_range=range(2, 7)) -> dict[int, float]:
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def cluster_position_group(df: pd.DataFrame, position: str, k: int, feature_cols: list[str] = FEATURE_COLS):
    group = df[df["position"] == position].copy()
    scaler = StandardScaler().fit(group[feature_cols])
    X = scaler.transform(group[feature_cols])
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    group["cluster"] = model.labels_

    # Centroid profile in original (unscaled) units — what actually
    # distinguishes each cluster, in numbers a person can read.
    centroids = group.groupby("cluster")[feature_cols].mean()
    return group, centroids
