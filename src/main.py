"""
CLI Entry Point for Algal Bloom Detection Pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algal_bloom.pipeline import AlgalBloomDetector


def main():
    parser = argparse.ArgumentParser(
        description="Automated Harmful Algal Bloom (HAB) Detection on Sentinel-2 Satellite Imagery"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to folder containing Sentinel-2 GeoTIFF bands (B2, B3, B4, B8, B11)"
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="outputs/algal_bloom_detection_results.png",
        help="Path where the 6-panel detection figure will be saved"
    )
    parser.add_argument(
        "--ndwi-threshold",
        type=float,
        default=0.2,
        help="NDWI threshold for water body extraction (default: 0.2)"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of K-Means clusters for water pixel grouping (default: 4)"
    )

    args = parser.parse_args()

    print(f"=== Algal Bloom Detection Pipeline ===")
    print(f"Input directory: {args.data_dir}")
    print(f"Target output plot: {args.output_plot}")
    print(f"NDWI threshold: {args.ndwi_threshold} | Clusters: {args.clusters}")

    try:
        detector = AlgalBloomDetector(args.data_dir)
        results = detector.run_detection(
            ndwi_threshold=args.ndwi_threshold,
            n_clusters=args.clusters
        )
        print("\n=== Detection Complete ===")
        print(json.dumps(results, indent=2))

        detector.visualize(output_path=args.output_plot)
        print(f"\nVisualization successfully saved to: {args.output_plot}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Input data error: {e}")
        print("Please ensure Sentinel-2 GeoTIFF bands are present in the specified directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
