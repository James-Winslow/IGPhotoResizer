# experiments/run_experiment.py
#
# Runs the full resize experiment across all images and methods.
#
# Supports both synthetic_v1 (flat folder) and synthetic_v2 (category subfolders)
# test set structures. Extracts category, aspect_ratio, and replicate from
# v2 filenames automatically for use in the mixed model analysis.
#
# Usage:
#   python experiments/run_experiment.py
#   python experiments/run_experiment.py --input-dir test_sets/synthetic_v2
#   python experiments/run_experiment.py --no-instagram
#   python experiments/run_experiment.py --skip-seam-carving

import os
import sys
import argparse
import traceback
from datetime import datetime

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods import METHODS
from src.metrics import compute_all_metrics
from src.instagram import PIPELINE_VARIANTS


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUT_DIR   = os.path.join(PROJECT_ROOT, "test_sets", "synthetic_v2")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TARGET_WIDTH  = 1080
TARGET_HEIGHT = 1080


def parse_filename(filename: str) -> tuple:
    """
    Extract category, aspect_ratio, and replicate from a v2 filename.

    v2 format: {category}_{aspect_ratio}_r{replicate}.png
    Examples:
        nebula_portrait_r1.png            -> ("nebula", "portrait", "1")
        architecture_very_wide_r3.png     -> ("architecture", "very_wide", "3")
        macro_biology_square_r2.png       -> ("macro_biology", "square", "2")
        abstract_texture_very_tall_r1.png -> ("abstract_texture", "very_tall", "1")

    v1 format: complex_image_{n}.jpg
        -> ("synthetic_v1", "unknown", "unknown")

    Returns ("unknown", "unknown", "unknown") for unrecognized formats.
    """
    stem = os.path.splitext(filename)[0]

    if stem.startswith("complex_image_"):
        return ("synthetic_v1", "unknown", "unknown")

    parts = stem.split("_")

    if len(parts) >= 3 and parts[-1].startswith("r") and parts[-1][1:].isdigit():
        replicate = parts[-1][1:]

        known_aspects = {"very_wide", "wide", "square", "portrait", "very_tall"}

        if len(parts) >= 4:
            two_word = f"{parts[-3]}_{parts[-2]}"
            if two_word in known_aspects:
                category = "_".join(parts[:-3])
                return (category, two_word, replicate)

        if parts[-2] in known_aspects:
            category = "_".join(parts[:-2])
            return (category, parts[-2], replicate)

    return ("unknown", "unknown", "unknown")


def discover_images(input_dir: str) -> list:
    """
    Find all images in input_dir.
    Supports flat folders (synthetic_v1) and category subfolders (synthetic_v2).
    Returns sorted list of absolute file paths.
    """
    extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []

    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if os.path.splitext(f)[1].lower() in extensions:
                image_paths.append(os.path.join(root, f))

    return sorted(image_paths)


def run_experiment(
    input_dir: str,
    results_dir: str,
    include_instagram: bool = True,
    methods_to_run: list = None,
    skip_seam_carving: bool = False
) -> pd.DataFrame:

    if methods_to_run is None:
        methods_to_run = list(METHODS.keys())

    if skip_seam_carving and "seam_carving_resize" in methods_to_run:
        print("Skipping seam_carving_resize (--skip-seam-carving flag set)")
        methods_to_run = [m for m in methods_to_run
                          if m != "seam_carving_resize"]

    image_paths = discover_images(input_dir)

    print(f"\n{'='*60}")
    print(f"IGPhotoResizer Experiment")
    print(f"{'='*60}")
    print(f"Input directory:    {input_dir}")
    print(f"Images found:       {len(image_paths)}")
    print(f"Methods:            {methods_to_run}")
    print(f"Instagram pipeline: {include_instagram}")
    print(f"{'='*60}\n")

    all_results   = []
    failed_images = []

    for image_index, image_path in enumerate(image_paths):
        image_filename = os.path.basename(image_path)
        category, aspect_ratio, replicate = parse_filename(image_filename)

        print(f"\n[{image_index + 1}/{len(image_paths)}] {image_filename}"
              f"  category={category}  aspect={aspect_ratio}  r={replicate}")

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
                preprocessed = method_fn(original, TARGET_WIDTH, TARGET_HEIGHT)

                preprocessing_metrics = compute_all_metrics(
                    original,
                    preprocessed,
                    method_name=method_name,
                    image_name=image_filename
                )
                preprocessing_metrics["pipeline"]      = "preprocessing_only"
                preprocessing_metrics["target_width"]  = TARGET_WIDTH
                preprocessing_metrics["target_height"] = TARGET_HEIGHT
                preprocessing_metrics["category"]      = category
                preprocessing_metrics["aspect_ratio"]  = aspect_ratio
                preprocessing_metrics["replicate"]     = replicate
                all_results.append(preprocessing_metrics)

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
                            instagram_metrics["pipeline"]      = variant_name
                            instagram_metrics["target_width"]  = TARGET_WIDTH
                            instagram_metrics["target_height"] = TARGET_HEIGHT
                            instagram_metrics["category"]      = category
                            instagram_metrics["aspect_ratio"]  = aspect_ratio
                            instagram_metrics["replicate"]     = replicate
                            all_results.append(instagram_metrics)

                        except Exception as e:
                            print(f"    ERROR in pipeline {variant_name}: {e}")
                            traceback.print_exc()

            except Exception as e:
                print(f"  ERROR in method {method_name}: {e}")
                traceback.print_exc()
                failed_images.append(f"{image_filename}::{method_name}")
                continue

    results_df = pd.DataFrame(all_results)

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
