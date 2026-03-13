# experiments/run_experiment.py
#
# Runs the full resize experiment across all images and methods.
#
# For each image in the test directory, applies every resize method,
# optionally runs the Instagram pipeline on top, computes all metrics,
# and saves results to a CSV in results/.
#
# Usage:
#   python experiments/run_experiment.py
#   python experiments/run_experiment.py --no-instagram
#   python experiments/run_experiment.py --input-dir frozen_test_images
#
# Design principles:
#   - One row per (image, method, pipeline_variant) combination
#   - No global state — everything flows through function arguments
#   - Seam carving is slow; progress is printed so you know it's alive
#   - Failed images are logged and skipped, not silently dropped

import os
import sys
import argparse
import traceback
from datetime import datetime

import pandas as pd
from PIL import Image

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods import METHODS
from src.metrics import compute_all_metrics
from src.instagram import PIPELINE_VARIANTS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUT_DIR   = os.path.join(PROJECT_ROOT, "frozen_test_images")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Instagram target dimensions for preprocessing step
TARGET_WIDTH  = 1080
TARGET_HEIGHT = 1080


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    input_dir: str,
    results_dir: str,
    include_instagram: bool = True,
    methods_to_run: list = None,
    skip_seam_carving: bool = False
) -> pd.DataFrame:
    """
    Run the full experiment and return results as a DataFrame.

    For each image:
        For each resize method:
            1. Apply resize method -> preprocessed image
            2. Compute metrics: preprocessed vs original (preprocessing quality)
            3. If include_instagram:
                For each Instagram pipeline variant:
                    Apply pipeline -> final image
                    Compute metrics: final vs original (end-to-end quality)

    This gives us two sets of measurements:
        - preprocessing_only: isolates the resize method's quality
        - after_instagram:    measures what survives the full pipeline

    The comparison between these two is the scientifically interesting part.
    """
    if methods_to_run is None:
        methods_to_run = list(METHODS.keys())

    if skip_seam_carving and "seam_carving_resize" in methods_to_run:
        print("Skipping seam_carving_resize (--skip-seam-carving flag set)")
        methods_to_run = [m for m in methods_to_run if m != "seam_carving_resize"]

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"\n{'='*60}")
    print(f"IGPhotoResizer Experiment")
    print(f"{'='*60}")
    print(f"Input directory:  {input_dir}")
    print(f"Images found:     {len(image_files)}")
    print(f"Methods:          {methods_to_run}")
    print(f"Instagram pipeline: {include_instagram}")
    print(f"{'='*60}\n")

    all_results = []
    failed_images = []

    for image_index, image_filename in enumerate(sorted(image_files)):
        print(f"\n[{image_index + 1}/{len(image_files)}] Processing: {image_filename}")
        image_path = os.path.join(input_dir, image_filename)

        try:
            original = Image.open(image_path).convert("RGB")
            print(f"  Loaded: {original.size[0]}x{original.size[1]}")
        except Exception as e:
            print(f"  ERROR loading {image_filename}: {e}")
            failed_images.append(image_filename)
            continue

        for method_name in methods_to_run:
            print(f"\n  --- Method: {method_name} ---")
            method_fn = METHODS[method_name]

            try:
                # Step 1: apply resize method
                preprocessed = method_fn(original, TARGET_WIDTH, TARGET_HEIGHT)

                # Step 2: measure preprocessing quality
                preprocessing_metrics = compute_all_metrics(
                    original,
                    preprocessed,
                    method_name=method_name,
                    image_name=image_filename
                )
                preprocessing_metrics["pipeline"] = "preprocessing_only"
                preprocessing_metrics["target_width"]  = TARGET_WIDTH
                preprocessing_metrics["target_height"] = TARGET_HEIGHT
                all_results.append(preprocessing_metrics)

                # Step 3: run Instagram pipeline variants
                if include_instagram:
                    for variant_name, pipeline_fn in PIPELINE_VARIANTS.items():
                        print(f"\n    Pipeline variant: {variant_name}")
                        try:
                            final_image = pipeline_fn(preprocessed)
                            instagram_metrics = compute_all_metrics(
                                original,
                                final_image,
                                method_name=method_name,
                                image_name=image_filename
                            )
                            instagram_metrics["pipeline"] = variant_name
                            instagram_metrics["target_width"]  = TARGET_WIDTH
                            instagram_metrics["target_height"] = TARGET_HEIGHT
                            all_results.append(instagram_metrics)

                        except Exception as e:
                            print(f"    ERROR in pipeline {variant_name}: {e}")
                            traceback.print_exc()

            except Exception as e:
                print(f"  ERROR in method {method_name}: {e}")
                traceback.print_exc()
                failed_images.append(f"{image_filename}::{method_name}")
                continue

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"experiment_results_{timestamp}.csv"
    output_path = os.path.join(results_dir, output_filename)
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Experiment complete")
    print(f"Total results rows: {len(results_df)}")
    print(f"Failed images:      {len(failed_images)}")
    if failed_images:
        print(f"Failed:             {failed_images}")
    print(f"Results saved to:   {output_path}")
    print(f"{'='*60}\n")

    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run IGPhotoResizer experiment across all methods and images"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing test images (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--no-instagram",
        action="store_true",
        help="Skip Instagram pipeline simulation (faster)"
    )
    parser.add_argument(
        "--skip-seam-carving",
        action="store_true",
        help="Skip seam carving method (much faster for quick runs)"
    )

    args = parser.parse_args()

    run_experiment(
        input_dir=args.input_dir,
        results_dir=DEFAULT_RESULTS_DIR,
        include_instagram=not args.no_instagram,
        skip_seam_carving=args.skip_seam_carving
    )