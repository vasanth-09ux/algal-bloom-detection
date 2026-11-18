"""
Unsupervised Spectral Clustering & Bloom Likelihood Scoring Module.
"""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.cluster import KMeans


def cluster_water_pixels(
    features: np.ndarray,
    water_mask: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[np.ndarray, KMeans | None]:
    """
    Applies K-Means clustering exclusively across valid water pixels.

    Args:
        features: 3D feature array of shape (H, W, num_features).
                  Typical features: [Blue, Green, Red, NIR, NDVI, FAI].
        water_mask: 2D boolean mask indicating water pixels.
        n_clusters: Number of clusters (default 4).
        random_state: Random seed for deterministic reproducibility.

    Returns:
        Tuple:
            - water_labels: 1D array of assigned cluster indices for water pixels.
            - kmeans: Fitted KMeans estimator instance, or None if no water pixels exist.
    """
    water_pixels = features[water_mask]

    if water_pixels.size == 0:
        return np.array([], dtype=int), None

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    water_labels = kmeans.fit_predict(water_pixels)

    return water_labels, kmeans


def rank_bloom_clusters(
    cluster_centers: np.ndarray,
    nir_idx: int = 3,
    green_idx: int = 1,
    blue_idx: int = 0,
    fai_idx: int = 5,
    nir_weight: float = 1.5,
    green_blue_weight: float = 1.0,
    fai_weight: float = 2.0,
    eps: float = 1e-6
) -> Tuple[int, np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Calculates composite bloom likelihood scores across cluster centroids.

    Heuristic Scoring Rationale:
        - Elevated NIR Reflectance: High biomass / surface scums reflect strongly in NIR.
        - High Green/Blue Ratio: Chlorophyll-a pigment strongly reflects green vs blue absorption.
        - High FAI: Floating algae produces pronounced positive FAI peaks above background water baseline.

    Formula:
        Score_k = nir_weight * NIR_k + green_blue_weight * (Green_k / (Blue_k + eps)) + fai_weight * FAI_k

    Args:
        cluster_centers: Array of shape (n_clusters, num_features) representing cluster centroids.
        nir_idx: Index of NIR band in feature vector.
        green_idx: Index of Green band in feature vector.
        blue_idx: Index of Blue band in feature vector.
        fai_idx: Index of FAI in feature vector.
        nir_weight: Weight assigned to NIR reflection (default 1.5).
        green_blue_weight: Weight assigned to Green/Blue ratio (default 1.0).
        fai_weight: Weight assigned to FAI (default 2.0).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Tuple:
            - bloom_cluster_idx: Index of cluster with highest likelihood of containing algal bloom.
            - bloom_scores: 1D array of bloom scores per cluster.
            - cluster_diagnostics: Dict mapping cluster index to individual component metrics.
    """
    n_clusters = cluster_centers.shape[0]
    bloom_scores = (
        cluster_centers[:, nir_idx] * nir_weight +
        (cluster_centers[:, green_idx] / (cluster_centers[:, blue_idx] + eps)) * green_blue_weight +
        cluster_centers[:, fai_idx] * fai_weight
    )

    bloom_cluster_idx = int(np.argmax(bloom_scores))

    diagnostics: Dict[int, Dict[str, float]] = {}
    for i in range(n_clusters):
        diagnostics[i] = {
            "nir_value": float(cluster_centers[i, nir_idx]),
            "green_blue_ratio": float(cluster_centers[i, green_idx] / (cluster_centers[i, blue_idx] + eps)),
            "fai_value": float(cluster_centers[i, fai_idx]),
            "bloom_score": float(bloom_scores[i])
        }

    return bloom_cluster_idx, bloom_scores, diagnostics
