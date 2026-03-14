# experiments/analyze_results.py
#
# Analyzes experiment results and produces summary statistics and visualizations.
# Designed to be run after experiments/run_experiment.py has produced a CSV.
#
# Usage:
#   python experiments/analyze_results.py
#   python experiments/analyze_results.py --results-file results/my_results.csv

import os
import sys
import argparse
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR      = os.path.join(PROJECT_ROOT, "results", "plots")

# Portfolio color palette (matches james-winslow.github.io)
PALETTE = {
    "simple_resize":  "#2ABFBF",   # teal
    "padding_resize": "#F5C842",   # yellow
    "background":     "#F7F5F0",   # warm off-white
    "text":           "#2C2C2C",
}

PIPELINE_ORDER = [
    "preprocessing_only",
    "instagram_standard",
    "instagram_high_quality",
    "instagram_low_quality",
    "instagram_square",
    "instagram_landscape",
]

PIPELINE_LABELS = {
    "preprocessing_only":    "Pre-processing\nonly",
    "instagram_standard":    "Instagram\nstandard (Q75)",
    "instagram_high_quality": "Instagram\nhigh (Q85)",
    "instagram_low_quality": "Instagram\nlow (Q70)",
    "instagram_square":      "Instagram\nsquare",
    "instagram_landscape":   "Instagram\nlandscape",
}


# ---------------------------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------------------------

def load_latest_results(results_dir: str, results_file: str = None) -> pd.DataFrame:
    if results_file:
        path = results_file
    else:
        csv_files = glob.glob(os.path.join(results_dir, "experiment_results_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No experiment result CSVs found in {results_dir}")
        path = sorted(csv_files)[-1]

    print(f"Loading results from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df['image'].nunique()} images, "
          f"{df['method'].nunique()} methods, {df['pipeline'].nunique()} pipelines")
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Use category and aspect_ratio if available (v2 test set)
    group_cols = ["method", "pipeline"]
    if "category" in df.columns and df["category"].nunique() > 1:
        group_cols = ["category", "aspect_ratio", "method", "pipeline"]

    summary = df.groupby(group_cols).agg(
        mean_mse    = ("mse",     "mean"),
        median_mse  = ("mse",     "median"),
        std_mse     = ("mse",     "std"),
        mean_psnr   = ("psnr_db", "mean"),
        median_psnr = ("psnr_db", "median"),
        mean_ssim   = ("ssim",    "mean"),
        median_ssim = ("ssim",    "median"),
        std_ssim    = ("ssim",    "std"),
        n           = ("image",   "count"),
    ).reset_index()
    return summary


def print_summary(summary: pd.DataFrame):
    print("\n" + "="*70)
    print("SUMMARY STATISTICS BY METHOD AND PIPELINE")
    print("="*70)

    for pipeline in PIPELINE_ORDER:
        subset = summary[summary["pipeline"] == pipeline]
        if subset.empty:
            continue
        print(f"\n--- {PIPELINE_LABELS.get(pipeline, pipeline)} ---")
        for _, row in subset.iterrows():
            print(f"  {row['method']:<25} "
                  f"MSE: {row['mean_mse']:>8.2f} Â± {row['std_mse']:>6.2f}  |  "
                  f"PSNR: {row['mean_psnr']:>6.2f} dB  |  "
                  f"SSIM: {row['mean_ssim']:.6f}")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired tests comparing simple_resize vs padding_resize for each pipeline.

    We use paired tests because each image is processed by both methods â€”
    the measurements are not independent. This is the same logic as a
    paired t-test in a crossover clinical trial.

    We run both a paired t-test (assumes normality) and a Wilcoxon signed-rank
    test (non-parametric) and report both. With n=50 images, the t-test is
    reasonably robust to mild non-normality by the CLT.
    """
    results = []
    methods = df["method"].unique()

    if len(methods) < 2:
        print("Only one method found â€” skipping pairwise tests")
        return pd.DataFrame()

    method_a = "simple_resize"
    method_b = "padding_resize"

    for pipeline in df["pipeline"].unique():
        subset = df[df["pipeline"] == pipeline]
        a = subset[subset["method"] == method_a].set_index("image")["ssim"]
        b = subset[subset["method"] == method_b].set_index("image")["ssim"]

        # Align on image names
        common_images = a.index.intersection(b.index)
        a_vals = a.loc[common_images].values
        b_vals = b.loc[common_images].values

        if len(a_vals) < 3:
            continue

        t_stat, t_p = ttest_rel(a_vals, b_vals)
        w_stat, w_p = wilcoxon(a_vals, b_vals)

        mean_diff = np.mean(a_vals - b_vals)

        results.append({
            "pipeline":        pipeline,
            "method_a":        method_a,
            "method_b":        method_b,
            "n_images":        len(common_images),
            "mean_ssim_a":     np.mean(a_vals),
            "mean_ssim_b":     np.mean(b_vals),
            "mean_diff_a_minus_b": mean_diff,
            "ttest_p":         t_p,
            "wilcoxon_p":      w_p,
            "significant_05":  (t_p < 0.05) and (w_p < 0.05),
        })

    return pd.DataFrame(results)


def print_statistical_tests(test_df: pd.DataFrame):
    print("\n" + "="*70)
    print("STATISTICAL TESTS: simple_resize vs padding_resize (SSIM)")
    print("Paired t-test and Wilcoxon signed-rank test")
    print("Positive mean_diff means simple_resize has higher SSIM")
    print("="*70)
    for _, row in test_df.iterrows():
        sig = "*** SIGNIFICANT" if row["significant_05"] else "not significant"
        print(f"\n{PIPELINE_LABELS.get(row['pipeline'], row['pipeline'])}")
        print(f"  Mean SSIM: simple={row['mean_ssim_a']:.6f}, "
              f"padding={row['mean_ssim_b']:.6f}, "
              f"diff={row['mean_diff_a_minus_b']:+.6f}")
        print(f"  t-test p={row['ttest_p']:.4f}, "
              f"Wilcoxon p={row['wilcoxon_p']:.4f}  [{sig}]")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def setup_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["background"],
        "axes.facecolor":    PALETTE["background"],
        "axes.edgecolor":    PALETTE["text"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "text.color":        PALETTE["text"],
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def plot_ssim_by_pipeline(df: pd.DataFrame, output_dir: str):
    """
    The main story plot: how does SSIM degrade through the Instagram pipeline
    for each method?
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    pipelines_present = [p for p in PIPELINE_ORDER if p in df["pipeline"].values]

    for method, color in [("simple_resize", PALETTE["simple_resize"]),
                           ("padding_resize", PALETTE["padding_resize"])]:
        method_data = df[df["method"] == method]
        means  = []
        errors = []
        for pipeline in pipelines_present:
            vals = method_data[method_data["pipeline"] == pipeline]["ssim"].values
            means.append(np.mean(vals))
            errors.append(np.std(vals) / np.sqrt(len(vals)))  # SEM

        x = range(len(pipelines_present))
        ax.plot(x, means, color=color, linewidth=2.5, marker="o",
                markersize=7, label=method.replace("_", " ").title())
        ax.fill_between(x,
                        [m - e for m, e in zip(means, errors)],
                        [m + e for m, e in zip(means, errors)],
                        color=color, alpha=0.15)

    ax.set_xticks(range(len(pipelines_present)))
    ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines_present],
                       fontsize=9)
    ax.set_ylabel("Mean SSIM (Â± SEM)", fontsize=11)
    ax.set_title("Image Quality Through the Instagram Pipeline\n"
                 "SSIM vs Original (higher = better)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=min(0.85, ax.get_ylim()[0] - 0.01))

    plt.tight_layout()
    path = os.path.join(output_dir, "ssim_by_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_mse_by_pipeline(df: pd.DataFrame, output_dir: str):
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pipelines_present = [p for p in PIPELINE_ORDER if p in df["pipeline"].values]

    for ax, method, color in zip(axes,
                                  ["simple_resize", "padding_resize"],
                                  [PALETTE["simple_resize"], PALETTE["padding_resize"]]):
        method_data = df[df["method"] == method]
        means = []
        for pipeline in pipelines_present:
            vals = method_data[method_data["pipeline"] == pipeline]["mse"].values
            means.append(np.mean(vals))

        bars = ax.bar(range(len(pipelines_present)), means, color=color, alpha=0.85)
        ax.set_xticks(range(len(pipelines_present)))
        ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines_present],
                           fontsize=8)
        ax.set_ylabel("Mean MSE", fontsize=11)
        ax.set_title(method.replace("_", " ").title(), fontsize=12, fontweight="bold")

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Mean Squared Error by Pipeline Stage", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "mse_by_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_image_scatter(df: pd.DataFrame, output_dir: str):
    """
    Scatter: for each image, plot simple_resize SSIM vs padding_resize SSIM
    after Instagram standard pipeline. Points above the diagonal = simple wins.
    Points below = padding wins.
    """
    setup_plot_style()
    pipeline = "instagram_standard"
    subset = df[df["pipeline"] == pipeline]

    simple  = subset[subset["method"] == "simple_resize"].set_index("image")["ssim"]
    padding = subset[subset["method"] == "padding_resize"].set_index("image")["ssim"]
    common  = simple.index.intersection(padding.index)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(simple.loc[common], padding.loc[common],
               color=PALETTE["simple_resize"], alpha=0.7, s=60, edgecolors="white")

    # Diagonal = equal performance
    lim_min = min(simple.loc[common].min(), padding.loc[common].min()) - 0.001
    lim_max = max(simple.loc[common].max(), padding.loc[common].max()) + 0.001
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color=PALETTE["text"], linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Simple Resize SSIM", fontsize=11)
    ax.set_ylabel("Padding Resize SSIM", fontsize=11)
    ax.set_title("Per-Image Quality After Instagram Pipeline\n"
                 "Points above diagonal = padding resize wins",
                 fontsize=12, fontweight="bold")

    n_simple_wins  = (simple.loc[common].values > padding.loc[common].values).sum()
    n_padding_wins = (padding.loc[common].values > simple.loc[common].values).sum()
    ax.text(0.05, 0.95,
            f"Simple wins: {n_simple_wins}/{len(common)}\n"
            f"Padding wins: {n_padding_wins}/{len(common)}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=PALETTE["background"], alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "per_image_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_aspect_ratio_vs_degradation(df: pd.DataFrame, output_dir: str):
    """
    Does aspect ratio predict how much padding_resize degrades after Instagram?
    Images with extreme aspect ratios have more border pixels, so they should
    degrade more.
    """
    setup_plot_style()

    padding_standard = df[
        (df["method"] == "padding_resize") &
        (df["pipeline"] == "instagram_standard")
    ].copy()

    padding_pre = df[
        (df["method"] == "padding_resize") &
        (df["pipeline"] == "preprocessing_only")
    ].set_index("image")[["ssim", "orig_width", "orig_height"]]

    padding_standard = padding_standard.set_index("image")
    padding_standard["ssim_pre"] = padding_pre["ssim"]
    padding_standard["ssim_degradation"] = (
        padding_standard["ssim_pre"] - padding_standard["ssim"]
    )
    padding_standard["aspect_ratio"] = (
        padding_pre["orig_width"] / padding_pre["orig_height"]
    )
    # Distance from square (1:1) â€” more extreme = more padding needed
    padding_standard["aspect_distance"] = abs(
        padding_standard["aspect_ratio"] - 1.0
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(padding_standard["aspect_distance"],
               padding_standard["ssim_degradation"],
               color=PALETTE["padding_resize"], alpha=0.7, s=60, edgecolors="white")

    # Fit a regression line
    x = padding_standard["aspect_distance"].values
    y = padding_standard["ssim_degradation"].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color=PALETTE["text"], linewidth=1.5,
            linestyle="--", alpha=0.7, label=f"Trend (slope={z[0]:.4f})")

    ax.set_xlabel("Aspect Ratio Distance from Square\n(|width/height - 1|)",
                  fontsize=11)
    ax.set_ylabel("SSIM Degradation\n(preprocessing_only â†’ instagram_standard)",
                  fontsize=11)
    ax.set_title("Does Aspect Ratio Predict Quality Loss?\nPadding Resize After Instagram",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    corr = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=PALETTE["background"], alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "aspect_ratio_vs_degradation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze IGPhotoResizer experiment results")
    parser.add_argument("--results-file", default=None,
                        help="Path to specific results CSV (default: latest in results/)")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_latest_results(RESULTS_DIR, args.results_file)

    summary = compute_summary(df)
    print_summary(summary)

    summary_path = os.path.join(RESULTS_DIR, "summary_statistics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to: {summary_path}")

    test_df = run_statistical_tests(df)
    if not test_df.empty:
        print_statistical_tests(test_df)
        test_path = os.path.join(RESULTS_DIR, "statistical_tests.csv")
        test_df.to_csv(test_path, index=False)
        print(f"\nStatistical tests saved to: {test_path}")

    print("\nGenerating plots...")
    plot_ssim_by_pipeline(df, PLOTS_DIR)
    plot_mse_by_pipeline(df, PLOTS_DIR)
    plot_image_scatter(df, PLOTS_DIR)
    plot_aspect_ratio_vs_degradation(df, PLOTS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
