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
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "results", "plots")

PALETTE = {
    "simple_resize":  "#2ABFBF",
    "padding_resize": "#F5C842",
    "seam_carving_resize": "#F27D9D",
    "background":     "#F7F5F0",
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
    "preprocessing_only":     "Pre-processing only",
    "instagram_standard":     "Instagram standard (Q75)",
    "instagram_high_quality": "Instagram high (Q85)",
    "instagram_low_quality":  "Instagram low (Q70)",
    "instagram_square":       "Instagram square",
    "instagram_landscape":    "Instagram landscape",
}

CATEGORY_ORDER = [
    "nebula", "gradient", "mountain",
    "forest", "coral", "abstract_texture",
    "architecture", "macro_biology",
    "synthetic_v1",
]

ASPECT_ORDER = ["very_wide", "wide", "square", "portrait", "very_tall", "unknown"]


# ---------------------------------------------------------------------------
# Load data
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

    has_v2 = "category" in df.columns and df["category"].nunique() > 1
    print(f"Loaded {len(df)} rows | "
          f"{df['image'].nunique()} images | "
          f"{df['method'].nunique()} methods | "
          f"{df['pipeline'].nunique()} pipelines | "
          f"v2 factorial={'yes' if has_v2 else 'no'}")
    return df


def is_v2(df: pd.DataFrame) -> bool:
    return "category" in df.columns and df["category"].nunique() > 1


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "pipeline"]
    if is_v2(df):
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
    print("\n" + "="*75)
    print("SUMMARY STATISTICS")
    print("="*75)

    for pipeline in PIPELINE_ORDER:
        subset = summary[summary["pipeline"] == pipeline]
        if subset.empty:
            continue
        print(f"\n--- {PIPELINE_LABELS.get(pipeline, pipeline)} ---")
        for _, row in subset.iterrows():
            prefix = ""
            if "category" in row.index:
                prefix = f"{row['category']:<20} {row['aspect_ratio']:<12} "
            print(f"  {prefix}{row['method']:<25} "
                  f"MSE: {row['mean_mse']:>8.2f} +/- {row['std_mse']:>6.2f}  |  "
                  f"PSNR: {row['mean_psnr']:>6.2f} dB  |  "
                  f"SSIM: {row['mean_ssim']:.6f}")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired t-test and Wilcoxon signed-rank test comparing simple_resize vs
    padding_resize for each pipeline. If v2 data, also runs tests per category.
    """
    results = []
    method_a = "simple_resize"
    method_b = "padding_resize"

    if method_a not in df["method"].values or method_b not in df["method"].values:
        print("Cannot run pairwise tests: need both simple_resize and padding_resize")
        return pd.DataFrame()

    # Overall test across all images
    for pipeline in df["pipeline"].unique():
        subset = df[df["pipeline"] == pipeline]
        a = subset[subset["method"] == method_a].set_index("image")["ssim"]
        b = subset[subset["method"] == method_b].set_index("image")["ssim"]
        common = a.index.intersection(b.index)

        if len(common) < 3:
            continue

        a_vals = a.loc[common].values
        b_vals = b.loc[common].values
        t_stat, t_p = ttest_rel(a_vals, b_vals)
        w_stat, w_p = wilcoxon(a_vals, b_vals)

        results.append({
            "scope":               "overall",
            "category":            "all",
            "pipeline":            pipeline,
            "n_images":            len(common),
            "mean_ssim_simple":    np.mean(a_vals),
            "mean_ssim_padding":   np.mean(b_vals),
            "mean_diff":           np.mean(a_vals - b_vals),
            "ttest_p":             t_p,
            "wilcoxon_p":          w_p,
            "significant_05":      (t_p < 0.05) and (w_p < 0.05),
        })

    # Per-category tests if v2
    if is_v2(df):
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            for pipeline in cat_df["pipeline"].unique():
                subset = cat_df[cat_df["pipeline"] == pipeline]
                a = subset[subset["method"] == method_a].set_index("image")["ssim"]
                b = subset[subset["method"] == method_b].set_index("image")["ssim"]
                common = a.index.intersection(b.index)

                if len(common) < 3:
                    continue

                a_vals = a.loc[common].values
                b_vals = b.loc[common].values
                t_stat, t_p = ttest_rel(a_vals, b_vals)
                w_stat, w_p = wilcoxon(a_vals, b_vals)

                results.append({
                    "scope":             "per_category",
                    "category":          category,
                    "pipeline":          pipeline,
                    "n_images":          len(common),
                    "mean_ssim_simple":  np.mean(a_vals),
                    "mean_ssim_padding": np.mean(b_vals),
                    "mean_diff":         np.mean(a_vals - b_vals),
                    "ttest_p":           t_p,
                    "wilcoxon_p":        w_p,
                    "significant_05":    (t_p < 0.05) and (w_p < 0.05),
                })

    return pd.DataFrame(results)


def print_statistical_tests(test_df: pd.DataFrame):
    print("\n" + "="*75)
    print("STATISTICAL TESTS: simple_resize vs padding_resize (SSIM)")
    print("Positive mean_diff means simple_resize has higher SSIM")
    print("="*75)

    overall = test_df[test_df["scope"] == "overall"]
    for _, row in overall.iterrows():
        sig = "*** SIGNIFICANT" if row["significant_05"] else "not significant"
        print(f"\n{PIPELINE_LABELS.get(row['pipeline'], row['pipeline'])}")
        print(f"  n={row['n_images']}  "
              f"simple={row['mean_ssim_simple']:.6f}  "
              f"padding={row['mean_ssim_padding']:.6f}  "
              f"diff={row['mean_diff']:+.6f}")
        print(f"  t-test p={row['ttest_p']:.4f}  "
              f"Wilcoxon p={row['wilcoxon_p']:.4f}  [{sig}]")

    per_cat = test_df[test_df["scope"] == "per_category"]
    if not per_cat.empty:
        print(f"\n{'='*75}")
        print("PER-CATEGORY TESTS (instagram_standard only)")
        print("="*75)
        ig_std = per_cat[per_cat["pipeline"] == "instagram_standard"]
        for _, row in ig_std.sort_values("mean_diff", ascending=False).iterrows():
            sig = "***" if row["significant_05"] else "n.s."
            print(f"  {row['category']:<22} "
                  f"diff={row['mean_diff']:+.4f}  "
                  f"n={row['n_images']}  [{sig}]")


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
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    pipelines_present = [p for p in PIPELINE_ORDER if p in df["pipeline"].values]
    methods = [m for m in ["simple_resize", "padding_resize", "seam_carving_resize"]
               if m in df["method"].values]

    for method in methods:
        color = PALETTE.get(method, "#888888")
        method_data = df[df["method"] == method]
        means, errors = [], []
        for pipeline in pipelines_present:
            vals = method_data[method_data["pipeline"] == pipeline]["ssim"].values
            means.append(np.mean(vals))
            errors.append(np.std(vals) / np.sqrt(len(vals)))

        x = range(len(pipelines_present))
        label = method.replace("_", " ").title()
        ax.plot(x, means, color=color, linewidth=2.5, marker="o",
                markersize=7, label=label)
        ax.fill_between(x,
                        [m - e for m, e in zip(means, errors)],
                        [m + e for m, e in zip(means, errors)],
                        color=color, alpha=0.15)

    ax.set_xticks(range(len(pipelines_present)))
    ax.set_xticklabels([PIPELINE_LABELS.get(p, p) for p in pipelines_present],
                       fontsize=9)
    ax.set_ylabel("Mean SSIM (+/- SEM)", fontsize=11)
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


def plot_ssim_by_category(df: pd.DataFrame, output_dir: str):
    """V2 only: SSIM after instagram_standard, broken out by category."""
    if not is_v2(df):
        return

    setup_plot_style()
    pipeline = "instagram_standard"
    subset = df[df["pipeline"] == pipeline]

    categories = [c for c in CATEGORY_ORDER if c in subset["category"].values]
    methods = [m for m in ["simple_resize", "padding_resize"]
               if m in subset["method"].values]

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, method in enumerate(methods):
        color = PALETTE.get(method, "#888888")
        means, errors = [], []
        for cat in categories:
            vals = subset[
                (subset["method"] == method) &
                (subset["category"] == cat)
            ]["ssim"].values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            errors.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, color=color, alpha=0.85,
                      label=method.replace("_", " ").title())
        ax.errorbar(x + offset, means, yerr=errors, fmt="none",
                    color=PALETTE["text"], capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean SSIM", fontsize=11)
    ax.set_title("Quality After Instagram Pipeline by Image Category\n"
                 "Instagram Standard (Q75)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ymin = ax.get_ylim()[0]
    ax.set_ylim(bottom=max(0.5, ymin - 0.02))
    plt.tight_layout()
    path = os.path.join(output_dir, "ssim_by_category.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_ssim_by_aspect_ratio(df: pd.DataFrame, output_dir: str):
    """V2 only: SSIM after instagram_standard, broken out by aspect ratio."""
    if not is_v2(df):
        return

    setup_plot_style()
    pipeline = "instagram_standard"
    subset = df[df["pipeline"] == pipeline]
    aspects = [a for a in ASPECT_ORDER if a in subset["aspect_ratio"].values]
    methods = [m for m in ["simple_resize", "padding_resize"]
               if m in subset["method"].values]

    x = np.arange(len(aspects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        color = PALETTE.get(method, "#888888")
        means, errors = [], []
        for asp in aspects:
            vals = subset[
                (subset["method"] == method) &
                (subset["aspect_ratio"] == asp)
            ]["ssim"].values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            errors.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, color=color, alpha=0.85,
               label=method.replace("_", " ").title())
        ax.errorbar(x + offset, means, yerr=errors, fmt="none",
                    color=PALETTE["text"], capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(aspects, fontsize=10)
    ax.set_ylabel("Mean SSIM", fontsize=11)
    ax.set_title("Quality After Instagram Pipeline by Aspect Ratio\n"
                 "Instagram Standard (Q75)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ymin = ax.get_ylim()[0]
    ax.set_ylim(bottom=max(0.5, ymin - 0.02))
    plt.tight_layout()
    path = os.path.join(output_dir, "ssim_by_aspect_ratio.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_per_image_scatter(df: pd.DataFrame, output_dir: str):
    setup_plot_style()
    pipeline = "instagram_standard"
    subset = df[df["pipeline"] == pipeline]
    simple  = subset[subset["method"] == "simple_resize"].set_index("image")["ssim"]
    padding = subset[subset["method"] == "padding_resize"].set_index("image")["ssim"]
    common  = simple.index.intersection(padding.index)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Color by category if available
    if is_v2(df):
        cat_map = subset[subset["method"] == "simple_resize"].set_index("image")["category"]
        categories = df["category"].unique()
        cmap = plt.cm.get_cmap("tab10", len(categories))
        cat_to_color = {c: cmap(i) for i, c in enumerate(categories)}
        colors = [cat_to_color.get(cat_map.get(img, "unknown"), "#888888")
                  for img in common]
        ax.scatter(simple.loc[common], padding.loc[common],
                   c=colors, alpha=0.7, s=60, edgecolors="white")
        handles = [plt.scatter([], [], color=cat_to_color[c], label=c, s=40)
                   for c in categories if c in cat_map.values]
        ax.legend(handles=handles, fontsize=8, loc="lower right",
                  title="Category", title_fontsize=8)
    else:
        ax.scatter(simple.loc[common], padding.loc[common],
                   color=PALETTE["simple_resize"], alpha=0.7, s=60,
                   edgecolors="white")

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
    padding_standard["ssim_pre"]         = padding_pre["ssim"]
    padding_standard["ssim_degradation"] = (padding_standard["ssim_pre"]
                                             - padding_standard["ssim"])
    padding_standard["aspect_ratio_val"] = (padding_pre["orig_width"]
                                             / padding_pre["orig_height"])
    padding_standard["aspect_distance"]  = abs(
        padding_standard["aspect_ratio_val"] - 1.0
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(padding_standard["aspect_distance"],
               padding_standard["ssim_degradation"],
               color=PALETTE["padding_resize"], alpha=0.7, s=60,
               edgecolors="white")

    x = padding_standard["aspect_distance"].values
    y = padding_standard["ssim_degradation"].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color=PALETTE["text"], linewidth=1.5,
            linestyle="--", alpha=0.7, label=f"Trend (slope={z[0]:.4f})")

    corr = np.corrcoef(x, y)[0, 1]
    ax.set_xlabel("Aspect Ratio Distance from Square\n(|width/height - 1|)",
                  fontsize=11)
    ax.set_ylabel("SSIM Degradation\n(preprocessing only to instagram standard)",
                  fontsize=11)
    ax.set_title("Does Aspect Ratio Predict Quality Loss?\nPadding Resize After Instagram",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
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
    parser = argparse.ArgumentParser(
        description="Analyze IGPhotoResizer experiment results"
    )
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
    plot_per_image_scatter(df, PLOTS_DIR)
    plot_aspect_ratio_vs_degradation(df, PLOTS_DIR)

    if is_v2(df):
        print("V2 dataset detected - generating category and aspect ratio plots...")
        plot_ssim_by_category(df, PLOTS_DIR)
        plot_ssim_by_aspect_ratio(df, PLOTS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
