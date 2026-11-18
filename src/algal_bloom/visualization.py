"""
Multi-Panel Spatial Visualization Module for Algal Bloom Detection.
Faithfully reproduces the 6-panel visualization from the original research notebook.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def enhance_true_color(true_color: np.ndarray, percentile: float = 98.0, eps: float = 1e-8) -> np.ndarray:
    """
    Applies percentile-based contrast stretching matching the original notebook:
        top = np.percentile(img, percentile)
        scaled = img / top
        return np.clip(scaled, 0, 1)
    """
    top = np.percentile(true_color, percentile)
    scaled = true_color / (top + eps)
    return np.clip(scaled, 0.0, 1.0)


def plot_detection_results(
    true_color: np.ndarray,
    water_mask: np.ndarray,
    bloom_mask: np.ndarray,
    fai: np.ndarray,
    ndvi: np.ndarray,
    nir: np.ndarray,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (20, 15),
    dpi: int = 300
) -> plt.Figure:
    """
    Renders the 6-panel diagnostic figure faithful to the original research notebook:
        Panel 1: True Color Image (B4-Red, B3-Green, B2-Blue with 98th-percentile enhancement).
        Panel 2: Water Mask (NDWI > 0.2 with 3x3 morphological opening and closing).
        Panel 3: Floating Algae Index (FAI) over water pixels only (RdYlGn colormap).
        Panel 4: Normalized Difference Vegetation Index (NDVI) over water pixels only (RdYlGn colormap).
        Panel 5: Near-Infrared (NIR) Reflectance over water pixels only (gray colormap).
        Panel 6: Final Unsupervised Anomaly Map (Blue: Clean Water, Red: Potential Bloom, Brown: Land).

    Args:
        true_color: Array of shape (H, W, 3) representing [Red, Green, Blue].
        water_mask: 2D boolean array of identified water bodies.
        bloom_mask: 2D boolean array of identified potential algal bloom pixels.
        fai: 2D array of Floating Algae Index values.
        ndvi: 2D array of Normalized Difference Vegetation Index values.
        nir: 2D array of Near-Infrared surface reflectance.
        save_path: Optional path to save high-resolution figure file.
        figsize: Figure dimensions (width, height) in inches.
        dpi: Output resolution dots per inch.

    Returns:
        plt.Figure: The rendered matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)

    # Panel 1: True Color Image
    ax1 = plt.subplot(231)
    enhanced_tc = enhance_true_color(true_color)
    plt.imshow(enhanced_tc)
    plt.title("True Color Image", fontsize=14)
    plt.axis("off")

    # Panel 2: Water Bodies Mask
    ax2 = plt.subplot(232)
    plt.imshow(water_mask, cmap="Blues")
    plt.title("Water Bodies", fontsize=14)
    legend_elements_water = [
        Patch(facecolor="blue", label="Water"),
        Patch(facecolor="white", edgecolor="black", label="Land")
    ]
    ax2.legend(handles=legend_elements_water, loc="upper right")
    plt.axis("off")

    # Panel 3: Floating Algae Index (FAI)
    ax3 = plt.subplot(233)
    masked_fai = np.ma.masked_array(fai, ~water_mask)
    im_fai = plt.imshow(masked_fai, cmap="RdYlGn")
    plt.colorbar(im_fai, ax=ax3, fraction=0.046, pad=0.04, label="FAI Value")
    plt.title("Floating Algae Index (Water Only)", fontsize=14)
    plt.axis("off")

    # Panel 4: NDVI
    ax4 = plt.subplot(234)
    masked_ndvi = np.ma.masked_array(ndvi, ~water_mask)
    im_ndvi = plt.imshow(masked_ndvi, cmap="RdYlGn")
    plt.colorbar(im_ndvi, ax=ax4, fraction=0.046, pad=0.04, label="NDVI Value")
    plt.title("NDVI (Water Only)", fontsize=14)
    plt.axis("off")

    # Panel 5: NIR Band
    ax5 = plt.subplot(235)
    masked_nir = np.ma.masked_array(nir, ~water_mask)
    im_nir = plt.imshow(masked_nir, cmap="gray")
    plt.colorbar(im_nir, ax=ax5, fraction=0.046, pad=0.04, label="NIR Reflectance")
    plt.title("NIR Band (Water Only)", fontsize=14)
    plt.axis("off")

    # Panel 6: Final Unsupervised Bloom Detection (Exact RGB colors from original notebook)
    ax6 = plt.subplot(236)
    bloom_viz = np.zeros((water_mask.shape[0], water_mask.shape[1], 3), dtype=float)
    # Clean Water = Blue [0, 0, 1]
    bloom_viz[water_mask & ~bloom_mask] = [0.0, 0.0, 1.0]
    # Potential Algal Bloom = Red [1, 0, 0]
    bloom_viz[bloom_mask] = [1.0, 0.0, 0.0]
    # Land = Brown [0.6, 0.4, 0.2]
    bloom_viz[~water_mask] = [0.6, 0.4, 0.2]

    plt.imshow(bloom_viz)
    plt.title("Algal Bloom Detection", fontsize=14)
    legend_elements_bloom = [
        Patch(facecolor="red", label="Potential Algal Bloom"),
        Patch(facecolor="blue", label="Clean Water"),
        Patch(facecolor=(0.6, 0.4, 0.2), label="Land")
    ]
    ax6.legend(handles=legend_elements_bloom, loc="upper right")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")

    return fig
