# src/test_single_image.py
#
# Test a single real image through all resize methods and the Instagram pipeline.
# Produces a side-by-side comparison grid with metrics.
#
# Usage:
#   python src/test_single_image.py --image path/to/your/photo.jpg
#   python src/test_single_image.py --image path/to/photo.jpg --no-instagram

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods import simple_resize, padding_resize, METHODS
from src.metrics import compute_all_metrics
from src.instagram import run_instagram_pipeline

# Portfolio palette
PALETTE = {
    "simple_resize":  "#2ABFBF",
    "padding_resize": "#F5C842",
    "background":     "#F7F5F0",
    "text":           "#2C2C2C",
    "accent":         "#F27D9D",
}

TARGET_WIDTH  = 1080
TARGET_HEIGHT = 1080


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    print(f"Loaded: {os.path.basename(path)} — {img.size[0]}×{img.size[1]} px")
    return img


def run_single_image_test(
    image_path: str,
    include_instagram: bool = True,
    output_dir: str = None
):
    image_name = os.path.basename(image_path)
    original   = load_image(image_path)

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "single_image_tests"
        )
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Run methods
    # ---------------------------------------------------------------------------
    print("\nRunning resize methods...")
    simple_result  = simple_resize(original, TARGET_WIDTH, TARGET_HEIGHT)
    padding_result = padding_resize(original, TARGET_WIDTH, TARGET_HEIGHT)

    results = {}

    print("\nComputing preprocessing metrics...")
    results["simple_resize"] = {
        "result":         simple_result,
        "metrics_pre":    compute_all_metrics(original, simple_result,
                                              "simple_resize", image_name),
        "metrics_ig":     None,
        "ig_image":       None,
    }
    results["padding_resize"] = {
        "result":         padding_result,
        "metrics_pre":    compute_all_metrics(original, padding_result,
                                              "padding_resize", image_name),
        "metrics_ig":     None,
        "ig_image":       None,
    }

    if include_instagram:
        print("\nRunning Instagram pipeline...")
        for method_name, data in results.items():
            ig_image = run_instagram_pipeline(data["result"], post_type="portrait")
            data["ig_image"]   = ig_image
            data["metrics_ig"] = compute_all_metrics(
                original, ig_image, method_name + "+instagram", image_name
            )

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------
    print("\n" + "="*65)
    print(f"RESULTS: {image_name}")
    print(f"Original dimensions: {original.size[0]}×{original.size[1]}")
    print("="*65)

    header = f"{'Metric':<12} {'Simple (pre)':>14} {'Padding (pre)':>14}"
    if include_instagram:
        header += f" {'Simple (IG)':>12} {'Padding (IG)':>13}"
    print(header)
    print("-" * len(header))

    for metric, label in [("ssim", "SSIM"), ("psnr_db", "PSNR (dB)"), ("mse", "MSE")]:
        s_pre = results["simple_resize"]["metrics_pre"][metric]
        p_pre = results["padding_resize"]["metrics_pre"][metric]

        if metric == "ssim":
            row = f"{label:<12} {s_pre:>14.6f} {p_pre:>14.6f}"
        elif metric == "psnr_db":
            row = f"{label:<12} {s_pre:>14.2f} {p_pre:>14.2f}"
        else:
            row = f"{label:<12} {s_pre:>14.4f} {p_pre:>14.4f}"

        if include_instagram:
            s_ig = results["simple_resize"]["metrics_ig"][metric]
            p_ig = results["padding_resize"]["metrics_ig"][metric]
            if metric == "ssim":
                row += f" {s_ig:>12.6f} {p_ig:>13.6f}"
            elif metric == "psnr_db":
                row += f" {s_ig:>12.2f} {p_ig:>13.2f}"
            else:
                row += f" {s_ig:>12.4f} {p_ig:>13.4f}"
        print(row)

    print("="*65)

    # ---------------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------------
    print("\nGenerating comparison grid...")

    n_cols = 5 if include_instagram else 3
    fig = plt.figure(figsize=(4 * n_cols, 8))
    fig.patch.set_facecolor(PALETTE["background"])

    gs = gridspec.GridSpec(2, n_cols, figure=fig,
                           hspace=0.4, wspace=0.05)

    def add_image_panel(ax, img, title, subtitle="", border_color=None):
        ax.imshow(np.array(img))
        ax.set_title(title, fontsize=9, fontweight="bold",
                     color=PALETTE["text"], pad=4)
        if subtitle:
            ax.text(0.5, -0.08, subtitle, transform=ax.transAxes,
                    ha="center", fontsize=7.5, color=PALETTE["text"])
        ax.axis("off")
        if border_color:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

    # Row 0: simple resize
    # Row 1: padding resize
    for row, (method_name, color, label) in enumerate([
        ("simple_resize",  PALETTE["simple_resize"],  "Simple Resize"),
        ("padding_resize", PALETTE["padding_resize"], "Padding Resize"),
    ]):
        data = results[method_name]
        m_pre = data["metrics_pre"]

        # Col 0: original (only show once, row 0)
        if row == 0:
            ax_orig = fig.add_subplot(gs[0, 0])
            add_image_panel(ax_orig, original,
                            "Original",
                            f"{original.size[0]}×{original.size[1]}")
            # blank for row 1 col 0
            ax_blank = fig.add_subplot(gs[1, 0])
            ax_blank.axis("off")
        else:
            pass

        # Col 1: preprocessed
        ax_pre = fig.add_subplot(gs[row, 1])
        pre_img = data["result"].image
        ssim_pre = m_pre["ssim"]
        psnr_pre = m_pre["psnr_db"]
        add_image_panel(
            ax_pre, pre_img,
            f"{label}",
            f"SSIM {ssim_pre:.4f} | PSNR {psnr_pre:.1f}dB",
            border_color=color
        )

        if include_instagram:
            # Col 2: after Instagram
            ax_ig = fig.add_subplot(gs[row, 2])
            ig_img  = data["ig_image"]
            m_ig    = data["metrics_ig"]
            ssim_ig = m_ig["ssim"]
            psnr_ig = m_ig["psnr_db"]
            add_image_panel(
                ax_ig, ig_img,
                f"{label} + Instagram",
                f"SSIM {ssim_ig:.4f} | PSNR {psnr_ig:.1f}dB",
                border_color=color
            )

            # Col 3: difference map (original vs after Instagram)
            ax_diff = fig.add_subplot(gs[row, 3])
            orig_arr = np.array(original.resize(ig_img.size,
                                Image.Resampling.LANCZOS), dtype=np.float32)
            ig_arr   = np.array(ig_img, dtype=np.float32)
            diff     = np.abs(orig_arr - ig_arr).mean(axis=2)
            diff_norm = diff / diff.max() if diff.max() > 0 else diff
            im = ax_diff.imshow(diff_norm, cmap="hot", vmin=0, vmax=1)
            ax_diff.set_title(f"Difference Map\n{label}",
                              fontsize=9, fontweight="bold",
                              color=PALETTE["text"], pad=4)
            ax_diff.axis("off")

    fig.suptitle(
        f"Resize Method Comparison: {image_name}\n"
        f"Original: {original.size[0]}×{original.size[1]}  →  "
        f"Target: {TARGET_WIDTH}×{TARGET_HEIGHT}",
        fontsize=12, fontweight="bold",
        color=PALETTE["text"], y=1.01
    )

    stem = os.path.splitext(image_name)[0]
    output_path = os.path.join(output_dir, f"{stem}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["background"])
    plt.close()
    print(f"\nComparison grid saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a single image through all resize methods"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the image file to test"
    )
    parser.add_argument(
        "--no-instagram", action="store_true",
        help="Skip Instagram pipeline simulation"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save results (default: results/single_image_tests/)"
    )
    args = parser.parse_args()

    run_single_image_test(
        image_path=args.image,
        include_instagram=not args.no_instagram,
        output_dir=args.output_dir
    )