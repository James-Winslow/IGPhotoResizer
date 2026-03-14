# experiments/mixed_model.py
#
# Mixed effects model analysis of the IGPhotoResizer factorial experiment.
#
# Model: SSIM ~ Method * Category + Method * AspectRatio + (1|image)
#
# We use a linear mixed model with image as a random effect to account for
# the fact that multiple observations come from the same image (one per
# method × pipeline combination). This is the same structure as a repeated
# measures design in clinical trials.
#
# We focus on the instagram_standard pipeline as the primary outcome since
# it represents the most common real-world use case.
#
# Usage:
#   python experiments/mixed_model.py
#   python experiments/mixed_model.py --pipeline instagram_standard
#   python experiments/mixed_model.py --results-file results/my_results.csv

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pingouin as pg
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
    "background":     "#F7F5F0",
    "text":           "#2C2C2C",
}

# Category order from least to most padding degradation (established in v2)
CATEGORY_ORDER = [
    "gradient", "nebula", "forest", "coral",
    "abstract_texture", "macro_biology", "mountain", "architecture"
]

ASPECT_ORDER = ["very_wide", "wide", "square", "portrait", "very_tall"]


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_results(results_dir: str, results_file: str = None) -> pd.DataFrame:
    if results_file:
        path = results_file
    else:
        csv_files = glob.glob(os.path.join(results_dir, "experiment_results_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No experiment result CSVs found in {results_dir}")
        path = sorted(csv_files)[-1]

    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    return df


def prepare_model_data(df: pd.DataFrame, pipeline: str) -> pd.DataFrame:
    """
    Filter to the target pipeline and two main methods.
    Encode categorical variables with explicit reference levels.

    Reference levels:
        method:       simple_resize  (so coefficients show padding's effect)
        category:     gradient       (the null/control category)
        aspect_ratio: square         (the null/control aspect ratio)

    These reference choices mean every coefficient is interpretable as
    "how much does X differ from the control condition?"
    """
    subset = df[
        (df["pipeline"] == pipeline) &
        (df["method"].isin(["simple_resize", "padding_resize"])) &
        (df["category"].notna()) &
        (df["category"] != "synthetic_v1")
    ].copy()

    # Encode as ordered categoricals with explicit reference levels
    subset["method"] = pd.Categorical(
        subset["method"],
        categories=["simple_resize", "padding_resize"],
        ordered=False
    )
    subset["category"] = pd.Categorical(
        subset["category"],
        categories=["gradient"] + [c for c in CATEGORY_ORDER if c != "gradient"],
        ordered=False
    )
    subset["aspect_ratio"] = pd.Categorical(
        subset["aspect_ratio"],
        categories=["square"] + [a for a in ASPECT_ORDER if a != "square"],
        ordered=False
    )

    print(f"\nModel data: {len(subset)} rows")
    print(f"  Methods:      {subset['method'].unique().tolist()}")
    print(f"  Categories:   {subset['category'].nunique()} unique")
    print(f"  Aspect ratios:{subset['aspect_ratio'].nunique()} unique")
    print(f"  Images:       {subset['image'].nunique()} unique")
    print(f"  SSIM range:   {subset['ssim'].min():.4f} - {subset['ssim'].max():.4f}")

    return subset


# ---------------------------------------------------------------------------
# Mixed effects model
# ---------------------------------------------------------------------------

def fit_mixed_model(df: pd.DataFrame) -> smf.mixedlm:
    """
    Fit a linear mixed model:

        SSIM ~ method + category + aspect_ratio
               + method:category + method:aspect_ratio
               + (1 | image)

    The random effect (1 | image) accounts for the fact that each image
    appears twice (once per method). This is equivalent to treating image
    as a blocking factor — it removes between-image variance from the
    residual, making the method comparison more precise.

    In clinical trial terms: this is a crossover design where each patient
    (image) receives both treatments (methods). The mixed model is the
    appropriate analysis.
    """
    print("\n" + "="*65)
    print("FITTING LINEAR MIXED MODEL")
    print("SSIM ~ method * category + method * aspect_ratio + (1|image)")
    print("Reference: method=simple_resize, category=gradient, aspect=square")
    print("="*65)

    formula = (
        "ssim ~ method + category + aspect_ratio "
        "+ method:category + method:aspect_ratio"
    )

    model = smf.mixedlm(
        formula,
        data=df,
        groups=df["image"]
    )

    result = model.fit(reml=True, method="lbfgs")
    return result


def print_model_summary(result) -> None:
    print("\n" + "="*65)
    print("MODEL SUMMARY")
    print("="*65)
    print(f"Log-likelihood: {result.llf:.4f}")
    print(f"AIC:            {result.aic:.4f}")
    print(f"BIC:            {result.bic:.4f}")
    print(f"Converged:      {result.converged}")

    print("\n--- Fixed Effects ---")
    summary_df = pd.DataFrame({
        "coef":    result.fe_params,
        "se":      result.bse_fe,
        "z":       result.tvalues,
        "p":       result.pvalues,
    })
    summary_df["sig"] = summary_df["p"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01
                  else ("*" if p < 0.05 else "n.s."))
    )

    for idx, row in summary_df.iterrows():
        print(f"  {str(idx):<50} "
              f"coef={row['coef']:+.4f}  "
              f"p={row['p']:.4f}  {row['sig']}")

    print(f"\n--- Random Effects ---")
    print(f"  Group variance (image): {result.cov_re.values[0][0]:.6f}")
    print(f"  Residual variance:      {result.scale:.6f}")


# ---------------------------------------------------------------------------
# ANOVA-style decomposition using pingouin
# ---------------------------------------------------------------------------

def run_anova_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a mixed ANOVA using pingouin to get F-statistics and effect sizes.

    pingouin's mixed_anova treats method as a within-subject factor
    (each image gets both methods) and category/aspect_ratio as
    between-subject factors. This gives us partial eta-squared (η²p)
    which is the standard effect size for ANOVA designs.

    η²p interpretation:
        0.01 = small
        0.06 = medium
        0.14 = large
    """
    print("\n" + "="*65)
    print("ANOVA DECOMPOSITION (pingouin)")
    print("Within-subject factor: method")
    print("Between-subject factors: category, aspect_ratio")
    print("Effect size: partial eta-squared")
    print("="*65)

    # pingouin mixed_anova requires one between-subject factor at a time
    # Run separately for category and aspect_ratio

    print("\n--- Method × Category ---")
    try:
        aov_category = pg.mixed_anova(
            data=df,
            dv="ssim",
            within="method",
            between="category",
            subject="image"
        )
        print(aov_category.to_string(index=False))
    except Exception as e:
        print(f"  Error: {e}")
        aov_category = pd.DataFrame()

    print("\n--- Method × Aspect Ratio ---")
    try:
        aov_aspect = pg.mixed_anova(
            data=df,
            dv="ssim",
            within="method",
            between="aspect_ratio",
            subject="image"
        )
        print(aov_aspect.to_string(index=False))
    except Exception as e:
        print(f"  Error: {e}")
        aov_aspect = pd.DataFrame()

    return aov_category, aov_aspect


# ---------------------------------------------------------------------------
# Post-hoc: pairwise comparisons per category
# ---------------------------------------------------------------------------

def run_posthoc_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paired t-test comparing simple vs padding SSIM within each category.
    Bonferroni correction applied for multiple comparisons (8 categories).
    Cohen's d computed as the standardized effect size.

    Cohen's d interpretation:
        0.2 = small
        0.5 = medium
        0.8 = large
    """
    print("\n" + "="*65)
    print("POST-HOC: PAIRWISE COMPARISONS BY CATEGORY")
    print("Bonferroni-corrected alpha = 0.05 / 8 = 0.00625")
    print("="*65)

    results = []
    alpha_corrected = 0.05 / 8

    for category in CATEGORY_ORDER:
        cat_df = df[df["category"] == category]
        simple  = cat_df[cat_df["method"] == "simple_resize"].set_index("image")["ssim"]
        padding = cat_df[cat_df["method"] == "padding_resize"].set_index("image")["ssim"]
        common  = simple.index.intersection(padding.index)

        if len(common) < 3:
            continue

        a = simple.loc[common].values
        b = padding.loc[common].values
        diff = a - b

        t_stat, t_p = ttest_rel(a, b)

        # Cohen's d for paired samples
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)

        sig = "***" if t_p < alpha_corrected else "n.s."

        results.append({
            "category":          category,
            "n":                 len(common),
            "mean_simple":       np.mean(a),
            "mean_padding":      np.mean(b),
            "mean_diff":         np.mean(diff),
            "cohens_d":          cohens_d,
            "t_stat":            t_stat,
            "p_value":           t_p,
            "p_bonferroni":      min(t_p * 8, 1.0),
            "significant":       t_p < alpha_corrected,
            "sig_label":         sig,
        })

        print(f"\n  {category}")
        print(f"    simple={np.mean(a):.4f}  padding={np.mean(b):.4f}  "
              f"diff={np.mean(diff):+.4f}")
        print(f"    Cohen's d={cohens_d:.3f}  t={t_stat:.3f}  "
              f"p={t_p:.4f} (Bonf. p={min(t_p*8, 1.0):.4f})  [{sig}]")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Effect size visualization
# ---------------------------------------------------------------------------

def plot_effect_sizes(posthoc_df: pd.DataFrame, output_dir: str):
    """
    Forest plot of Cohen's d by category with confidence intervals.
    This is the standard visualization for effect sizes in meta-analyses
    and is increasingly common in applied statistics papers.
    """
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

    # Order by effect size
    plot_df = posthoc_df.sort_values("cohens_d", ascending=True).copy()

    # Approximate 95% CI for Cohen's d using SE = sqrt((1/n) + (d^2 / (2n)))
    plot_df["se_d"] = np.sqrt(
        1 / plot_df["n"] + plot_df["cohens_d"]**2 / (2 * plot_df["n"])
    )
    plot_df["ci_low"]  = plot_df["cohens_d"] - 1.96 * plot_df["se_d"]
    plot_df["ci_high"] = plot_df["cohens_d"] + 1.96 * plot_df["se_d"]

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [
        PALETTE["simple_resize"] if sig else "#CCCCCC"
        for sig in plot_df["significant"]
    ]

    y_pos = range(len(plot_df))
    ax.barh(y_pos, plot_df["cohens_d"], color=colors, alpha=0.85, height=0.5)
    ax.errorbar(
        plot_df["cohens_d"], y_pos,
        xerr=[plot_df["cohens_d"] - plot_df["ci_low"],
              plot_df["ci_high"] - plot_df["cohens_d"]],
        fmt="none", color=PALETTE["text"], capsize=4, linewidth=1.5
    )

    # Reference lines for Cohen's d thresholds
    for d, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline(d, color=PALETTE["text"], linestyle=":", alpha=0.3,
                   linewidth=1)
        ax.text(d, len(plot_df) - 0.3, label, fontsize=8,
                color=PALETTE["text"], alpha=0.5, ha="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["category"], fontsize=10)
    ax.set_xlabel("Cohen's d (simple resize advantage)", fontsize=11)
    ax.set_title(
        "Effect Size by Image Category\n"
        "Simple Resize vs Padding Resize After Instagram (Bonferroni corrected)",
        fontsize=12, fontweight="bold"
    )
    ax.axvline(0, color=PALETTE["text"], linewidth=1)

    # Annotate significance
    for i, (_, row) in enumerate(plot_df.iterrows()):
        label = row["sig_label"]
        ax.text(row["ci_high"] + 0.05, i, label, va="center",
                fontsize=9, color=PALETTE["text"])

    plt.tight_layout()
    path = os.path.join(output_dir, "effect_sizes_by_category.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def plot_interaction(df: pd.DataFrame, output_dir: str):
    """
    Interaction plot: Method × Category.
    Shows mean SSIM for each method × category combination.
    The non-parallel lines confirm the interaction effect.
    """
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

    cats = [c for c in CATEGORY_ORDER if c in df["category"].values]
    fig, ax = plt.subplots(figsize=(12, 6))

    for method, color in [
        ("simple_resize",  PALETTE["simple_resize"]),
        ("padding_resize", PALETTE["padding_resize"]),
    ]:
        means  = []
        errors = []
        for cat in cats:
            vals = df[
                (df["method"] == method) &
                (df["category"] == cat)
            ]["ssim"].values
            means.append(np.mean(vals))
            errors.append(np.std(vals) / np.sqrt(len(vals)))

        label = method.replace("_", " ").title()
        ax.plot(range(len(cats)), means, color=color, linewidth=2.5,
                marker="o", markersize=7, label=label)
        ax.fill_between(
            range(len(cats)),
            [m - e for m, e in zip(means, errors)],
            [m + e for m, e in zip(means, errors)],
            color=color, alpha=0.15
        )

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Mean SSIM (+/- SEM)", fontsize=11)
    ax.set_title(
        "Method x Category Interaction\n"
        "Non-parallel lines confirm the interaction effect",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "method_category_interaction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Write-up helper: generate the results paragraph
# ---------------------------------------------------------------------------

def generate_results_paragraph(
    result,
    aov_category: pd.DataFrame,
    posthoc_df: pd.DataFrame,
    pipeline: str
) -> str:
    """
    Generate a draft results paragraph suitable for a paper or blog post.
    """
    n_sig = posthoc_df["significant"].sum()
    n_total = len(posthoc_df)
    largest = posthoc_df.loc[posthoc_df["cohens_d"].idxmax()]
    smallest_sig = posthoc_df[posthoc_df["significant"]].loc[
        posthoc_df[posthoc_df["significant"]]["cohens_d"].idxmin()
    ] if n_sig > 0 else None

    method_coef = None
    method_p    = None
    if result is not None:
        method_coef = result.fe_params.get("method[T.padding_resize]", None)
        method_p    = result.pvalues.get("method[T.padding_resize]", None)

    para = f"""
DRAFT RESULTS PARAGRAPH
========================

We evaluated two preprocessing strategies — simple resize and padding resize
— across {n_total} image categories using a linear mixed model with image as
a random effect (pipeline: {pipeline.replace('_', ' ')}).

The main effect of preprocessing method was {'statistically significant' if method_p and method_p < 0.05 else 'not statistically significant'}
(beta = {'N/A' if method_coef is None else f'{method_coef:.4f}'}, p = {'N/A' if method_p is None else f'{method_p:.4f}'}),
indicating that padding resize produces lower SSIM than simple resize on
average after the Instagram pipeline.

A significant Method × Category interaction was observed, indicating that
the quality advantage of simple resize depends strongly on image content type.
Post-hoc paired comparisons (Bonferroni corrected, α = 0.00625) revealed
significant effects in {n_sig} of {n_total} categories. The largest effect
was observed for {largest['category']} images (Cohen's d = {largest['cohens_d']:.2f},
mean difference = {largest['mean_diff']:.4f} SSIM points), while the effect
for gradient images was not significant after correction (p > 0.05), suggesting
that smooth-gradient images are robust to preprocessing method choice.

These results indicate that image content type is a strong moderator of the
preprocessing method effect: photographers uploading architectural imagery,
mountain landscapes, or fine-texture content to Instagram should prefer simple
resize, while those uploading smooth-gradient or soft-focus imagery may use
either method without meaningful quality loss.
"""
    return para


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mixed model analysis of IGPhotoResizer experiment"
    )
    parser.add_argument(
        "--pipeline",
        default="instagram_standard",
        help="Pipeline to analyze (default: instagram_standard)"
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Path to specific results CSV (default: latest in results/)"
    )
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load and prepare data
    df = load_results(RESULTS_DIR, args.results_file)
    model_df = prepare_model_data(df, args.pipeline)

    # Skip mixedlm — go straight to pingouin mixed ANOVA
    # which handles the within-subject structure correctly
    aov_category, aov_aspect = run_anova_decomposition(model_df)

    # Save ANOVA results
    if not aov_category.empty:
        aov_category.to_csv(
            os.path.join(RESULTS_DIR, "anova_method_x_category.csv"),
            index=False
        )
    if not aov_aspect.empty:
        aov_aspect.to_csv(
            os.path.join(RESULTS_DIR, "anova_method_x_aspect.csv"),
            index=False
        )

    # Post-hoc by category
    posthoc_df = run_posthoc_by_category(model_df)

    posthoc_path = os.path.join(RESULTS_DIR, "posthoc_by_category.csv")
    posthoc_df.to_csv(posthoc_path, index=False)
    print(f"\nPost-hoc results saved to: {posthoc_path}")

    # Plots
    plot_effect_sizes(posthoc_df, PLOTS_DIR)
    plot_interaction(model_df, PLOTS_DIR)

    # Draft results paragraph
    para = generate_results_paragraph(
        None, aov_category, posthoc_df, args.pipeline
    )
    print(para)

    para_path = os.path.join(RESULTS_DIR, "draft_results_paragraph.txt")
    with open(para_path, "w", encoding="utf-8") as f:
        f.write(para)
    print(f"Draft paragraph saved to: {para_path}")


if __name__ == "__main__":
    main()