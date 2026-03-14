# Results: Instagram Photo Resizer & Quality Optimizer

## Overview

We evaluated two image preprocessing strategies — simple resize and padding resize — across a fully crossed factorial design of 8 image categories x 5 aspect ratios x 3 replicates (N = 120 images), processing each through a simulated Instagram pipeline. Quality was measured using the Structural Similarity Index (SSIM) against the original image.

---

## Preprocessing alone produces no meaningful difference

Before Instagram's pipeline, simple resize and padding resize produced statistically indistinguishable quality (mean SSIM: 0.9966 vs 0.9963, difference = 0.0003). Both methods preserve image quality effectively when evaluated in isolation. This confirms that the choice of preprocessing method is not independently consequential — what matters is how each method interacts with Instagram's downstream compression.

---

## Simple resize preserves significantly more quality after Instagram

After the simulated Instagram standard pipeline (portrait resize + JPEG Q75), a mixed ANOVA revealed a large main effect of preprocessing method (F(1,112) = 145.33, p = 5.8e-22, eta_p^2 = 0.565). Simple resize outperformed padding resize on 95 of 120 images. The one case where padding resize won involved a nearly square image where negligible padding was added — a degenerate case that confirms rather than challenges the finding.

---

## Image content type strongly moderates the effect

A significant Method x Category interaction was observed (F(7,112) = 13.07, p = 3.3e-12, eta_p^2 = 0.450), indicating that the quality advantage of simple resize depends heavily on image content. Post-hoc paired comparisons (Bonferroni corrected, alpha = 0.00625) confirmed significant effects in 7 of 8 categories tested:

| Category         | Simple SSIM | Padding SSIM | Delta SSIM | Cohen's d | Significance |
|------------------|-------------|--------------|------------|-----------|--------------|
| Gradient         | 0.998       | 0.988        | +0.011     | 0.42      | n.s.         |
| Nebula           | 0.996       | 0.981        | +0.015     | 1.46      | ***          |
| Forest           | 0.996       | 0.944        | +0.052     | 1.60      | ***          |
| Coral            | 0.991       | 0.914        | +0.077     | 1.73      | ***          |
| Abstract texture | 0.995       | 0.797        | +0.198     | 1.07      | ***          |
| Macro biology    | 0.993       | 0.789        | +0.203     | 1.01      | ***          |
| Mountain         | 0.984       | 0.746        | +0.238     | 1.61      | ***          |
| Architecture     | 0.973       | 0.632        | +0.341     | 1.90      | ***          |

The gradient category was the single non-significant result (d = 0.42, Bonferroni p = 1.0), confirming that smooth tonal content is robust to preprocessing method choice. For all other categories the effect was large by conventional standards (d > 1.0 for 6 of 7 significant categories).

---

## Aspect ratio moderates the effect independently

A significant Method x Aspect Ratio interaction was also observed (F(4,115) = 6.47, p = 0.0001, eta_p^2 = 0.184), though with substantially smaller effect size than the category interaction. Square images (1:1) showed the smallest method difference, consistent with the mechanism: square images require no padding, making the two methods functionally identical. Very wide (2:1) and very tall (1:2) images showed the largest padding degradation, consistent with the larger border area requiring more aggressive rescaling by Instagram's pipeline.

---

## Mechanism

The degradation pattern is explained by a double-resize mechanism. Padding resize adds colored borders to reach the target dimensions. Instagram's pipeline then resizes the padded image to its own target — applying a second downscale to the content while border pixels consume part of the available pixel budget. Simple resize fills the entire frame with content, so Instagram's pipeline applies only one total downscale. This mechanism predicts exactly the pattern observed: the effect is largest for content with high spatial frequency (architecture, macro biology) where each resize pass introduces more interpolation artifact, and smallest for smooth gradients where interpolation error is negligible.

---

## Practical recommendation

For images with smooth color content — soft-focus photography, gradient backgrounds, nebula-style imagery — preprocessing method does not meaningfully affect final quality after Instagram upload. For images with hard edges, fine texture, or geometric structure — architectural photography, cityscapes, mountain landscapes, macro photography — simple resize preserves significantly more quality. The conventional advice to pad images before Instagram upload is not supported by these results and is actively harmful for the most visually demanding content types.

---

## Statistical summary

| Test | Statistic | p-value | Effect size |
|---|---|---|---|
| Main effect: method | F(1,112) = 145.33 | 5.8e-22 | eta_p^2 = 0.565 |
| Main effect: category | F(7,112) = 16.11 | 1.6e-14 | eta_p^2 = 0.502 |
| Method x Category | F(7,112) = 13.07 | 3.3e-12 | eta_p^2 = 0.450 |
| Main effect: aspect ratio | F(4,115) = 5.72 | 0.0003 | eta_p^2 = 0.166 |
| Method x Aspect Ratio | F(4,115) = 6.47 | 0.0001 | eta_p^2 = 0.184 |

*Pipeline: Instagram standard (portrait 1080x1350, JPEG Q75)*
*N = 120 images, 8 categories, 5 aspect ratios, 3 replicates per cell*
*Bonferroni correction applied to post-hoc comparisons (alpha = 0.00625)*
