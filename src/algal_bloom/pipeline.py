"""
End-to-End Algal Bloom Detection Pipeline.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import os
import numpy as np
import rasterio

from .indices import compute_ndwi, compute_ndvi, compute_fai
from .water_mask import generate_water_mask
from .clustering import cluster_water_pixels, rank_bloom_clusters
from .visualization import plot_detection_results


class AlgalBloomDetector:
    """
    Executes the complete unsupervised HAB detection workflow on Sentinel-2 GeoTIFF bands.
    """

    REQUIRED_BANDS = {
        "blue": "sentinel2_B2.tif",
        "green": "sentinel2_B3.tif",
        "red": "sentinel2_B4.tif",
        "nir": "sentinel2_B8.tif",
        "swir": "sentinel2_B11.tif"
    }

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.bands_data: Dict[str, np.ndarray] = {}
        self.water_mask: Optional[np.ndarray] = None
        self.bloom_mask: Optional[np.ndarray] = None
        self.diagnostics: Dict[str, Any] = {}

    def load_bands(self, scale_factor: float = 10000.0) -> None:
        """Loads and normalizes Sentinel-2 GeoTIFFs to surface reflectance [0.0, 1.0]."""
        for band_key, filename in self.REQUIRED_BANDS.items():
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing required band file: {path}")

            with rasterio.open(path) as src:
                raw_data = src.read(1).astype(float)
                self.bands_data[band_key] = raw_data / scale_factor

    def run_detection(
        self,
        ndwi_threshold: float = 0.2,
        n_clusters: int = 4,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Executes water delineation, spectral feature extraction, K-Means clustering,
        and bloom likelihood ranking.
        """
        if not self.bands_data:
            self.load_bands()

        blue = self.bands_data["blue"]
        green = self.bands_data["green"]
        red = self.bands_data["red"]
        nir = self.bands_data["nir"]
        swir = self.bands_data["swir"]

        # 1. Compute NDWI and generate morphologically filtered water mask
        ndwi = compute_ndwi(green, nir)
        self.water_mask = generate_water_mask(ndwi, threshold=ndwi_threshold)

        # 2. Compute Vegetation & Floating Algae Indices
        ndvi = compute_ndvi(nir, red)
        fai = compute_fai(red, nir, swir)

        # 3. Assemble Multi-Spectral Feature Vector
        features = np.stack([blue, green, red, nir, ndvi, fai], axis=-1)

        # 4. Cluster water pixels and rank clusters by bloom score
        water_labels, kmeans = cluster_water_pixels(
            features, self.water_mask, n_clusters=n_clusters, random_state=random_state
        )

        self.bloom_mask = np.zeros_like(self.water_mask, dtype=bool)

        if kmeans is not None:
            cluster_centers = kmeans.cluster_centers_
            bloom_idx, bloom_scores, cluster_diagnostics = rank_bloom_clusters(cluster_centers)
            self.bloom_mask[self.water_mask] = (water_labels == bloom_idx)

            self.diagnostics = {
                "n_clusters": n_clusters,
                "bloom_cluster_index": bloom_idx,
                "cluster_diagnostics": cluster_diagnostics,
                "fai_statistics": {
                    "min": float(np.nanmin(fai)),
                    "max": float(np.nanmax(fai)),
                    "mean": float(np.nanmean(fai))
                },
                "total_water_pixels": int(np.sum(self.water_mask)),
                "detected_bloom_pixels": int(np.sum(self.bloom_mask)),
                "bloom_percentage_of_water": float(
                    (np.sum(self.bloom_mask) / (np.sum(self.water_mask) + 1e-8)) * 100.0
                )
            }
        else:
            self.diagnostics = {"error": "No water pixels detected in image scene."}

        return self.diagnostics

    def visualize(self, output_path: Optional[str | Path] = None):
        """Generates the 6-panel spatial detection visualization."""
        if self.water_mask is None or self.bloom_mask is None:
            raise RuntimeError("Must call run_detection() before visualizing.")

        true_color = np.stack([
            self.bands_data["red"],
            self.bands_data["green"],
            self.bands_data["blue"]
        ], axis=-1)

        ndvi = compute_ndvi(self.bands_data["nir"], self.bands_data["red"])
        fai = compute_fai(self.bands_data["red"], self.bands_data["nir"], self.bands_data["swir"])

        return plot_detection_results(
            true_color=true_color,
            water_mask=self.water_mask,
            bloom_mask=self.bloom_mask,
            fai=fai,
            ndvi=ndvi,
            nir=self.bands_data["nir"],
            save_path=output_path
        )
