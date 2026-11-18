"""
Algal Bloom Detection Package.
Modular satellite remote sensing and machine learning pipeline for detecting Harmful Algal Blooms (HABs).
"""

from .indices import compute_ndwi, compute_ndvi, compute_fai
from .water_mask import generate_water_mask
from .clustering import cluster_water_pixels, rank_bloom_clusters
from .visualization import plot_detection_results
from .pipeline import AlgalBloomDetector

__all__ = [
    "compute_ndwi",
    "compute_ndvi",
    "compute_fai",
    "generate_water_mask",
    "cluster_water_pixels",
    "rank_bloom_clusters",
    "plot_detection_results",
    "AlgalBloomDetector",
]

__version__ = "1.0.0"
