# Sentinel-2 Imagery Data Guide

This directory holds the multi-spectral Sentinel-2 Level-2A GeoTIFF raster files required for the Algal Bloom Detection pipeline.

## Required Spectral Bands

The detection and indexing algorithms expect the following 5 individual GeoTIFF files:

| Band Identifier | File Name | Spatial Resolution | Central Wavelength ($\lambda$) | Purpose in Pipeline |
| :--- | :--- | :---: | :---: | :--- |
| **Band 2 (Blue)** | `sentinel2_B2.tif` | 10 m | 490 nm | Multi-spectral feature clustering, True-color RGB |
| **Band 3 (Green)** | `sentinel2_B3.tif` | 10 m | 560 nm | NDWI calculation, True-color RGB |
| **Band 4 (Red)** | `sentinel2_B4.tif` | 10 m | 665 nm | NDVI, FAI baseline interpolation, True-color RGB |
| **Band 8 (NIR)** | `sentinel2_B8.tif` | 10 m | 842 nm | NDWI, NDVI, FAI peak reflection |
| **Band 11 (SWIR-1)** | `sentinel2_B11.tif`| 20 m (resampled) | 1610 nm | FAI baseline correction |

*(Optional: `sentinel2_B12.tif` - Band 12 / SWIR-2 at 2190 nm).*

---

## Data Sourcing & Acquisition

Due to GitHub repository size limits and individual region requirements, raw GeoTIFF files are excluded from git tracking via `.gitignore`. You can acquire imagery via either of the following two routes:

### Option A: Automated Download via Google Earth Engine (Recommended)
1. Ensure you have an authenticated Google Cloud project with Google Earth Engine API enabled.
2. Set your environment variable:
   ```bash
   export GEE_PROJECT_ID="your-gcp-project-id"
   ```
3. Use the GEE collection script or notebook (`notebooks/algal_bloom_analysis.ipynb`) to download bands directly for your custom Region of Interest (ROI):
   ```python
   from src.algal_bloom.gee_collector import GEECollector
   import ee

   collector = GEECollector()
   roi = ee.Geometry.BBox(80.0, 15.0, 86.0, 20.0)
   collector.download_all_bands(region=roi, output_dir="data", scale=100)
   ```

### Option B: Manual Ingestion via Copernicus Data Space
1. Visit the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).
2. Search and download the relevant Sentinel-2 L2A tile for your temporal window (<20% cloud cover).
3. Extract bands B02, B03, B04, B08, and B11 into this `data/` folder, matching the expected filenames above.

---

## Sample Study Area Metadata
- **Region:** Coastal Bay of Bengal / Andhra Pradesh coast
- **Coordinates:** Lat 15.0°N to 20.0°N, Lon 80.0°E to 86.0°E (Center: `17.7009°N, 83.2770°E`)
- **Temporal Filter:** 2022-01-01 to 2022-01-31
- **Sensor:** Multispectral Instrument (MSI) on Copernicus Sentinel-2
- **Processing Level:** Level-2A Bottom-Of-Atmosphere (BOA) Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`)
