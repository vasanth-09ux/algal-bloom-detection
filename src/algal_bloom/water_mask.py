"""
Water Body Delineation and Morphological Filtering Module.
"""

import numpy as np
from scipy import ndimage


def generate_water_mask(
    ndwi: np.ndarray,
    threshold: float = 0.2,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Delineates surface water bodies from NDWI using empirical thresholding
    and binary morphological noise filtering.

    Steps:
        1. Initial mask: NDWI > threshold (default 0.2).
        2. Morphological opening (erosion followed by dilation) to remove isolated noise pixels.
        3. Morphological closing (dilation followed by erosion) to bridge small gaps and holes.

    Args:
        ndwi: 2D array of Normalized Difference Water Index values.
        threshold: Floating point threshold for water classification (default 0.2).
        kernel_size: Square structuring element dimension (default 3x3).

    Returns:
        np.ndarray: Boolean mask where True indicates validated water pixels.
    """
    initial_mask = ndwi > threshold

    # Define structuring element
    struct = np.ones((kernel_size, kernel_size), dtype=bool)

    # Apply morphological opening to suppress speckle noise
    opened = ndimage.binary_opening(initial_mask, structure=struct)

    # Apply morphological closing to bridge internal discontinuities
    filtered_mask = ndimage.binary_closing(opened, structure=struct)

    return filtered_mask
