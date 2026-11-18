"""
Google Earth Engine (GEE) Sentinel-2 Satellite Ingestion Module.
Provides secure, parameterized utilities to query, filter, composite,
and download Sentinel-2 MSI surface reflectance GeoTIFFs.
"""

from typing import List, Tuple, Optional
import os
import math
from pathlib import Path
import requests

try:
    import ee
    import geemap
    HAS_GEE = True
except ImportError:
    HAS_GEE = False


class GEECollector:
    """
    Automates Sentinel-2 Level-2A imagery retrieval via Google Earth Engine.
    """

    def __init__(self, project_id: Optional[str] = None):
        """
        Initializes GEE connection without hardcoded credentials.

        Args:
            project_id: Google Cloud project ID with GEE enabled.
                        Defaults to environment variable GEE_PROJECT_ID.
        """
        if not HAS_GEE:
            raise ImportError(
                "Google Earth Engine dependencies are missing. Install with: pip install earthengine-api geemap"
            )

        self.project_id = project_id or os.environ.get("GEE_PROJECT_ID")
        self._authenticate_and_initialize()

    def _authenticate_and_initialize(self) -> None:
        """Initializes Earth Engine session."""
        try:
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
        except Exception:
            ee.Authenticate()
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()

    def get_sentinel_layer(
        self,
        region: "ee.Geometry",
        start_date: str = "2022-01-01",
        end_date: str = "2022-01-31",
        cloud_percentage_max: int = 20
    ) -> "ee.Image":
        """
        Queries and returns a temporal mean surface reflectance composite.
        """
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_percentage_max))
        )
        return collection.mean()

    def download_band(
        self,
        image: "ee.Image",
        band_name: str,
        region: "ee.Geometry",
        output_dir: str | Path,
        scale: int = 100,
        max_pixels: int = 25000000
    ) -> bool:
        """
        Downloads a specific Sentinel-2 band GeoTIFF with automated tiling if needed.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        band_img = image.select(band_name)

        bounds = region.bounds().getInfo()["coordinates"][0]
        min_x = min(pt[0] for pt in bounds)
        max_x = max(pt[0] for pt in bounds)
        min_y = min(pt[1] for pt in bounds)
        max_y = max(pt[1] for pt in bounds)

        width_deg = max_x - min_x
        height_deg = max_y - min_y
        approx_pixels = abs(width_deg * (1000 / scale)) * abs(height_deg * (1000 / scale))

        if approx_pixels > max_pixels:
            tiles_needed = math.ceil(approx_pixels / max_pixels)
            tiles_per_side = math.ceil(math.sqrt(tiles_needed))
            w_step = width_deg / tiles_per_side
            h_step = height_deg / tiles_per_side

            for i in range(tiles_per_side):
                for j in range(tiles_per_side):
                    t_min_x = min_x + (i * w_step)
                    t_max_x = min_x + ((i + 1) * w_step)
                    t_min_y = min_y + (j * h_step)
                    t_max_y = min_y + ((j + 1) * h_step)
                    sub_geom = ee.Geometry.Rectangle([t_min_x, t_min_y, t_max_x, t_max_y])

                    url = band_img.getDownloadURL({
                        "scale": scale,
                        "region": sub_geom,
                        "format": "GEO_TIFF",
                        "crs": "EPSG:4326",
                        "filePerBand": True
                    })
                    resp = requests.get(url, stream=True)
                    if resp.status_code == 200:
                        tile_path = out_dir / f"sentinel2_{band_name}_tile_{i}_{j}.tif"
                        with open(tile_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                f.write(chunk)
            return True
        else:
            url = band_img.getDownloadURL({
                "scale": scale,
                "region": region,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
                "filePerBand": True
            })
            resp = requests.get(url, stream=True)
            if resp.status_code == 200:
                out_path = out_dir / f"sentinel2_{band_name}.tif"
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            return False

    def download_all_bands(
        self,
        region: "ee.Geometry",
        output_dir: str | Path,
        bands: Optional[List[str]] = None,
        scale: int = 100
    ) -> List[str]:
        """Downloads all standard analysis bands."""
        if bands is None:
            bands = ["B2", "B3", "B4", "B8", "B11", "B12"]

        image = self.get_sentinel_layer(region)
        successes = []
        for b in bands:
            if self.download_band(image, b, region, output_dir, scale=scale):
                successes.append(b)
        return successes
