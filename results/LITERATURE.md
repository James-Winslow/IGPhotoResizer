# Literature Review: Instagram Photo Resizer & Quality Optimizer

This document covers the academic and empirical foundations of the project
across four areas: image quality metrics, content-aware resizing, social media
compression pipelines, and related statistical methods.

---

## 1. Image Quality Metrics

### 1.1 The SSIM Paper (Primary Citation)

**Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).**
Image quality assessment: From error visibility to structural similarity.
*IEEE Transactions on Image Processing*, 13(4), 600–612.
https://doi.org/10.1109/TIP.2003.819861

This is the foundational paper for SSIM and is required reading and citation
for any work using the metric. Key points:

- SSIM was motivated by the failure of MSE and PSNR to correlate with human
  perception. Two images with identical MSE can have dramatically different
  perceived quality.
- The metric decomposes similarity into three components: luminance (mean
  comparison), contrast (standard deviation comparison), and structure
  (normalized cross-correlation — essentially Pearson's r between patches).
- Computed over an 11x11 Gaussian-weighted sliding window. This local
  computation is what makes SSIM sensitive to spatial structure.
- Validated against human subjective ratings on JPEG and JPEG2000 compressed
  images — exactly the distortion type relevant to Instagram compression.
- The paper has over 50,000 citations (Google Scholar) and received the
  IEEE Signal Processing Society Best Paper Award (2009) and Sustained
  Impact Award (2016). Authors received a Primetime Engineering Emmy Award
  in 2015 for SSIM's adoption by the television industry.

**Relevance to this project:** SSIM is our primary quality metric.
The paper's validation on JPEG compression makes it directly applicable.
The finding that SSIM responds differently to different distortion types
(blurring, noise, compression) underpins our category interaction result —
categories with hard edges (architecture) are more sensitive to compression
artifacts than smooth categories (gradient).

**Citation format (APA):**
Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
Image quality assessment: From error visibility to structural similarity.
IEEE Transactions on Image Processing, 13(4), 600–612.

---

### 1.2 Multi-Scale SSIM

**Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003).**
Multiscale structural similarity for image quality assessment.
*Proc. IEEE Asilomar Conference on Signals, Systems and Computers.*

MS-SSIM computes SSIM at multiple spatial scales via iterative downsampling,
weighting luminance at the coarsest scale and contrast/structure at all scales.
Shown to outperform single-scale SSIM for certain conditions, particularly
larger images and mixed distortion types.

**Relevance:** A natural next metric to add to the project. MS-SSIM would
capture how quality degrades at multiple viewing distances — relevant to
Instagram where images are viewed at different resolutions depending on device.

---

### 1.3 3-SSIM and Edge Sensitivity

**Chen, G. H., Yang, C. L., & Xie, S. L. (2006).**
Gradient-based structural similarity for image quality assessment.
*Proc. IEEE International Conference on Image Processing.*

3-SSIM weights edge, texture, and smooth regions separately, with proposed
weights of 0.5 for edges and 0.25 each for texture and smooth regions.
Notably, 1/0/0 weighting (edges only) correlates most strongly with
subjective quality ratings, suggesting edge regions dominate human perception
of image quality.

**Relevance:** Directly explains our category interaction result. Architecture
and macro biology images are dominated by edges and textures; gradient images
are dominated by smooth regions. 3-SSIM's edge-weighting scheme predicts
exactly our observed ordering: architecture degrades most, gradient least.
This is worth explicitly citing in the discussion section.

---

### 1.4 LPIPS (Planned Future Metric)

**Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).**
The unreasonable effectiveness of deep features as a perceptual metric.
*Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).*
arXiv:1801.03924

LPIPS (Learned Perceptual Image Patch Similarity) uses deep neural network
feature distances as a perceptual similarity metric. Key findings:

- Features from AlexNet, VGG, and SqueezeNet trained on ImageNet classification
  all outperform SSIM and PSNR on human perceptual similarity judgments.
- Perceptual similarity is an emergent property of deep visual representations,
  not specific to any architecture or training objective.
- LPIPS is available as a pip-installable Python package: pip install lpips
- Uses PyTorch; inputs must be normalized to [-1, 1].

**Relevance:** LPIPS is the state-of-the-art perceptual metric and directly
addresses SSIM's limitation of being grayscale-only and not capturing
high-level semantic distortion. Adding LPIPS to the metric suite is the
single most impactful methodological improvement for v3. The key question
is whether LPIPS would show the same category interaction pattern as SSIM,
or reveal a different ordering — that comparison would itself be a publishable
finding.

**Citation format (APA):**
Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).
The unreasonable effectiveness of deep features as a perceptual metric.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (pp. 586–595).

---

## 2. Content-Aware Resizing

### 2.1 Seam Carving (Primary Citation)

**Avidan, S., & Shamir, A. (2007).**
Seam carving for content-aware image resizing.
*ACM Transactions on Graphics*, 26(3), Article 10.
https://doi.org/10.1145/1275808.1276390
Presented at SIGGRAPH 2007.

The foundational paper for seam carving. Key contributions:

- Defines a seam as an optimal 8-connected path of pixels (top-to-bottom or
  left-to-right) where optimality is defined by an energy function.
- Energy function: gradient magnitude (L1 or L2 norm of image derivatives).
  Low energy = uniform regions. High energy = edges, texture.
- Dynamic programming finds the minimum-energy seam in O(H x W) time.
  This is the DP recurrence implemented in our methods.py.
- Seams can be removed (shrink) or inserted (expand).
- Adobe licensed the technology for Photoshop CS4 (Content Aware Scaling).
- 18,490 downloads on ACM DL; widely implemented in GIMP, ImageMagick, etc.

**Relevance:** This is the algorithm we implemented as our third resize method.
The paper's energy function (gradient magnitude) is exactly what we use in
compute_energy_map(). Must cite in methods section.

**Citation format (APA):**
Avidan, S., & Shamir, A. (2007). Seam carving for content-aware image
resizing. ACM Transactions on Graphics, 26(3), Article 10.
https://doi.org/10.1145/1275808.1276390

---

### 2.2 Improved Seam Carving (Forward Energy)

**Rubinstein, M., Shamir, A., & Avidan, S. (2008).**
Improved seam carving for video retargeting.
*ACM Transactions on Graphics*, 27(3).

Introduces "forward energy" — an improved energy function that considers
the energy that will be created after seam removal, not just the current
energy. Reduces visible artifacts in areas of smooth gradients where
backward energy fails.

**Relevance:** Our current implementation uses backward energy (original
Avidan 2007). Forward energy would reduce artifacts in our gradient and
nebula categories. A planned improvement for v3.

---

### 2.3 Multi-Operator Retargeting

**Rubinstein, M., Gutierrez, D., Sorkine, O., & Shamir, A. (2009).**
A comparative study of image retargeting.
*ACM Transactions on Graphics*, 28(5).

Compares seam carving, cropping, scaling, and combination approaches
("multi-operator") on a dataset of 80 images with human preference ratings.
Key finding: no single method dominates — the best method depends on image
content, consistent with our interaction effect finding.

**Relevance:** The most directly relevant prior comparison study to our work.
Should be discussed in related work section. Their finding that content
moderates method effectiveness is exactly what we found, and they reach it
through a different (subjective) methodology.

---

## 3. Social Media Compression Pipelines

### 3.1 Community Empirical Estimates (Non-Academic)

The specific compression parameters used by Instagram are not publicly
documented. The following are the best available empirical estimates
from the photography community:

**Instagram pipeline (best current estimates):**
- Target dimensions: 1080 x 1080 (square), 1080 x 1350 (portrait, 4:5),
  1080 x 566 (landscape, 1.91:1)
- JPEG quality: estimated 70–85%, most sources converge on 75–80%
- Carousel behavior: first image sets aspect ratio for all subsequent images
- Oversized images (> 1080px wide) are downscaled before compression
- Desktop upload may apply less compression than mobile app upload
- All images recompressed regardless of input quality or dimensions

**Key sources:**
- iformat.io/blog/instagram-image-sizes-2026: Tested empirically, recommends
  JPG at 85-95% quality, confirms carousel aspect ratio locking behavior
- Photo Taco Podcast (phototacopodcast.com): 80+ hours of empirical testing
  across Facebook, Instagram, Twitter; recommends 77% quality for multi-platform
- Skylum blog: Recommends JPEG at 75-85% quality

**Gap in academic literature:** There is no peer-reviewed empirical study
of Instagram's specific compression pipeline parameters. Laghari et al. (2018)
studied Facebook, WeChat, Tumblr and Twitter but not Instagram specifically.
This is a gap our empirical verification experiment (planned for v3) would help fill.

---

### 3.2 Social Media Image Compression QoE Study

**Laghari, A. A., et al. (2018).**
Assessment of quality of experience (QoE) of image compression in social
cloud computing.
*IOS Press / International Journal of Intelligent Engineering and Systems.*

Subjective QoE study comparing image compression across Facebook, WeChat,
Tumblr, and Twitter. Key finding: Facebook and Twitter compress images less
aggressively than WeChat and Tumblr; both received acceptable user satisfaction
ratings.

**Relevance:** The only peer-reviewed study directly measuring social media
platform compression effects on perceived image quality. Does not include
Instagram. Provides methodological precedent for our empirical verification
approach. The subjective QoE methodology (human raters) complements our
objective metric approach.

**Citation format (APA):**
Laghari, A. A., et al. (2018). Assessment of quality of experience (QoE)
of image compression in social cloud computing. Journal of Intelligent
Engineering and Systems.

---

## 4. Related Statistical Methods

### 4.1 Mixed ANOVA and Repeated Measures Designs

Our analysis uses a mixed ANOVA (pingouin implementation) with:
- Within-subject factor: preprocessing method (each image processed by both)
- Between-subject factors: image category, aspect ratio
- Effect size: partial eta-squared (η²p)

**Key references for this methodology:**

Field, A. (2013). *Discovering statistics using IBM SPSS statistics*
(4th ed.). Sage Publications.
— Standard reference for mixed ANOVA; covers sphericity, Greenhouse-Geisser
correction, and interpretation of interaction effects.

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences*
(2nd ed.). Lawrence Erlbaum Associates.
— Standard reference for Cohen's d interpretation (0.2 small, 0.5 medium,
0.8 large) and power analysis.

**Planned additions for v3:**
- Mauchly's test of sphericity (currently not reported)
- Greenhouse-Geisser epsilon correction if sphericity violated
- Omega-squared (ω²) as less-biased alternative to η²p
- Prospective power analysis based on observed effect sizes

---

## 5. Methodological Comparisons and Gaps

### What exists in the literature:

| Topic | Status |
|---|---|
| SSIM as image quality metric | Well established, 50k+ citations |
| PSNR as image quality metric | Standard baseline, widely used |
| LPIPS as perceptual metric | State of the art since 2018 |
| Seam carving algorithm | Well established (Avidan 2007) |
| Social media image compression QoE | One study (Laghari 2018), no Instagram |
| Instagram-specific compression parameters | Community estimates only, no peer review |
| Preprocessing effects on Instagram quality | No prior academic work found |
| Content type as moderator of compression | No prior work in social media context |

### What this project contributes:

1. **First controlled comparison** of preprocessing strategies specifically
   for Instagram multi-photo posts
2. **Content type as a moderator** of preprocessing method effects —
   not previously studied in social media compression context
3. **Factorial experimental design** establishing that image category
   (not just aspect ratio) is the primary driver of quality differences
4. **Empirical pipeline verification** (planned v3) — would be the first
   peer-reviewed measurement of Instagram's actual compression parameters

---

## 6. Suggested Citation List for Paper

**Must cite:**
1. Wang et al. (2004) — SSIM
2. Avidan & Shamir (2007) — Seam carving
3. Zhang et al. (2018) — LPIPS (when added in v3)
4. Laghari et al. (2018) — Social media compression QoE

**Should cite:**
5. Wang et al. (2003) — MS-SSIM
6. Rubinstein et al. (2009) — Multi-operator retargeting comparison
7. Cohen (1988) — Effect size interpretation
8. Field (2013) — Mixed ANOVA methodology

**Consider citing:**
9. Chen et al. (2006) — 3-SSIM / edge sensitivity (explains our category result)
10. Rubinstein et al. (2008) — Forward energy seam carving (planned improvement)

---

## 7. Empirical Verification Plan (v3)

Before claiming our simulation accurately represents Instagram's pipeline,
we need empirical ground truth. Proposed experiment:

**Protocol:**
1. Generate 10 controlled synthetic test images with known measurable properties
   (gradients, checkerboards, known frequency content)
2. Upload each to Instagram as a single post and as positions 1 and 2 in a
   two-image carousel post
3. Download the processed versions via browser developer tools (Network tab,
   filter for .jpg responses) or instaloader Python library
4. Compare downloaded images against originals using SSIM, PSNR, MSE
5. Fit a JPEG quality model to estimate Instagram's actual compression level
6. Compare against our Q75 simulation — measure simulation accuracy

**Tool:** instaloader (pip install instaloader) — Python library for
downloading your own Instagram content programmatically.

**Expected output:** A table showing actual vs simulated pipeline parameters,
and a revised instagram.py with empirically validated parameters.

**This experiment takes approximately one afternoon** and is the single most
important step before submitting for publication.

---

*Last updated: March 2026*
*Status: Literature review complete for v2 publication. LPIPS and
forward energy seam carving planned for v3.*
