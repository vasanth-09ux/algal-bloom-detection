"""
Spectral Indices Calculation Module.
Implements Normalized Difference Water Index (NDWI), Normalized Difference
Vegetation Index (NDVI), and Floating Algae Index (FAI) for Sentinel-2 imagery.
"""

from typing import Dict
import numpy as np

# Nominal central wavelengths in nanometers for Copernicus Sentinel-2 MSI
SENTINEL2_WAVELENGTHS: Dict[str, float] = {
    "red": 665.0,   # Band 4
    "nir": 842.0,   # Band 8
    "swir": 1610.0  # Band 11 (SWIR-1)
}


def compute_ndwi(green: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Computes the Normalized Difference Water Index (NDWI) according to McFeeters (1996).

    Formula:
        NDWI = (Green - NIR) / (Green + NIR)

    Args:
        green: Surface reflectance array for Green band (Sentinel-2 B3).
        nir: Surface reflectance array for NIR band (Sentinel-2 B8).
        eps: Epsilon to prevent division by zero.

    Returns:
        np.ndarray: Computed NDWI array with values bounded in [-1.0, 1.0].
    """
    denominator = green + nir + eps
    return (green - nir) / denominator


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Computes the Normalized Difference Vegetation Index (NDVI) according to Rouse et al. (1974).

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir: Surface reflectance array for NIR band (Sentinel-2 B8).
        red: Surface reflectance array for Red band (Sentinel-2 B4).
        eps: Epsilon to prevent division by zero.

    Returns:
        np.ndarray: Computed NDVI array with values bounded in [-1.0, 1.0].
    """
    denominator = nir + red + eps
    return (nir - red) / denominator


def compute_fai(
    red: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    wavelengths: Dict[str, float] | None = None
) -> np.ndarray:
    """
    Computes the Floating Algae Index (FAI) according to Hu (2009).

    Formula:
        lambda_ratio = (lambda_NIR - lambda_Red) / (lambda_SWIR - lambda_Red)
        Baseline_NIR = Red + (SWIR - Red) * lambda_ratio
        FAI = NIR - Baseline_NIR

    Args:
        red: Surface reflectance array for Red band (Sentinel-2 B4).
        nir: Surface reflectance array for NIR band (Sentinel-2 B8).
        swir: Surface reflectance array for SWIR-1 band (Sentinel-2 B11).
        wavelengths: Central wavelengths mapping (defaults to Sentinel-2 specs).

    Returns:
        np.ndarray: Computed Floating Algae Index (FAI) array.
    """
    if wavelengths is None:
        wavelengths = SENTINEL2_WAVELENGTHS

    lambda_red = wavelengths["red"]
    lambda_nir = wavelengths["nir"]
    lambda_swir = wavelengths["swir"]

    # Interpolation factor based on wavelength differences
    ratio = (lambda_nir - lambda_red) / (lambda_swir - lambda_red)

    # Linear baseline interpolation across NIR from Red and SWIR
    baseline_nir = red + (swir - red) * ratio

    return nir - baseline_nir
