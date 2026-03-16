# IGPhotoResizer: Roadmap and Future Work

This document captures the complete planned trajectory of the project,
ordered by phase. Each phase builds on the previous and produces standalone
publishable results.

Current status: v2 experiment complete. Synthetic factorial results documented.
Paused for real-world data collection (see PHOTO_PROTOCOL.md).

---

## Phase Summary

| Phase | Description | Status |
|---|---|---|
| v1 | Proof of concept, evaluation bug found | Complete |
| v2 | Factorial experiment, 8 categories, mixed ANOVA | Complete |
| v3 | Real images, seam carving optimization, color metrics | In progress |
| v4 | ML pipeline simulator, preprocessing optimizer | Planned |
| v5 | Computational textbook, full paper | Planned |

---

## v3: Real Images and Method Improvements

### 3.1 Empirical Instagram Verification

**Why this is required before publication:**
Our Instagram simulation uses Q75 JPEG + portrait resize, based on community
estimates. No peer-reviewed study has measured Instagram's actual compression
parameters. A skeptical reviewer can legitimately question whether our
simulation reflects reality. Empirical verification is the single most
important step before submitting for publication.

**Protocol:** See PHOTO_PROTOCOL.md for the complete field photography
and upload/download protocol.

**What to measure:**
- Actual JPEG quality factor used by Instagram (recoverable from EXIF or
  by fitting a JPEG model to the downloaded file)
- Whether compression differs between single posts and carousel positions
- Whether compression differs between position 1 and position 2+ in a carousel
- Whether desktop upload applies different compression than mobile
- Whether the "High Quality Uploads" setting changes the output

**Expected output:** A revised instagram.py with empirically validated
parameters, and a table in RESULTS.md showing simulation accuracy vs ground truth.

**Tool:** instaloader (pip install instaloader) for programmatic download
of your own Instagram content. Never screenshot — always download the file.

---

### 3.2 Seam Carving: NumPy Vectorization

**Current state:** Pure Python loop implementation. O(H x W x N_seams) time.
Takes overnight for 120 images.

**Target state:** NumPy vectorized DP. Same algorithm, 50-100x faster.
Full 120-image experiment feasible in under an hour.

**The key optimization:** The DP forward pass recurrence:

    M[i,j] = energy[i,j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])

can be computed row-by-row using np.roll and np.minimum rather than
nested Python loops. The backtracking similarly. No algorithmic change —
pure implementation speedup.

**Files to modify:** src/methods.py — the seam_carving_resize function
and its helpers compute_energy_map, compute_seam_dp, find_seam.

---

### 3.3 Saliency-Based Energy Function

**Current state:** Energy = gradient magnitude (Sobel operator).
Treats all edges as equally important regardless of semantic content.

**Problem:** Fails on portraits and images where the important content
is semantically defined rather than edge-defined. A face against a complex
background will have the face removed before the background.

**Proposed energy function:**

    energy = alpha * gradient_magnitude
           + beta * (1 - face_detection_confidence)
           + gamma * (1 - saliency_map)

where:
- gradient_magnitude: current implementation (keep)
- face_detection_confidence: OpenCV Haar cascades or DNN face detector
- saliency_map: pre-trained saliency model (e.g. ITTI, BMS, or deep saliency)

**Parameters alpha, beta, gamma:** to be tuned experimentally.
Start with equal weights (0.33 each) and optimize on a held-out set.

**Expected impact:** Large improvement on portrait and subject-focused
photography. Minimal impact on architecture and abstract texture.

---

### 3.4 Forward Energy Seam Carving

**Current state:** Backward energy (Avidan & Shamir 2007).
Minimizes energy of removed pixels.

**Problem:** Creates artifacts in smooth regions (gradients, sky, water)
because backward energy fails to account for the energy introduced
by juxtaposing previously non-adjacent pixels.

**Improvement:** Forward energy (Rubinstein et al. 2008).
Minimizes the energy that will be created by the removal, not the
energy of the removed pixel itself. Reduces staircase artifacts.

**Reference:**
Rubinstein, M., Shamir, A., & Avidan, S. (2008).
Improved seam carving for video retargeting.
ACM Transactions on Graphics, 27(3).

---

### 3.5 Real Image Test Set (Unsplash + Field Photography)

**Synthetic images:** Useful for controlled experiments but reviewers will
always ask about real photographs. The synthetic results are necessary but
not sufficient for publication.

**Two complementary real image sets:**

**Set A: Curated Unsplash images**
- 8 categories x 5 images each = 40 images
- Selected to match category definitions (architecture = buildings with
  hard geometric edges, mountain = landscape with horizontal bands, etc.)
- Download script: src/download_unsplash.py (to be written)
- License: Unsplash license permits research use

**Set B: Field photography (primary)**
- Personal photographs taken specifically for this experiment
- Exact originals available for comparison (no intermediate processing)
- See PHOTO_PROTOCOL.md for what to shoot and how
- This set provides the strongest ground truth because you have the
  uncompressed original directly from the camera

**Why field photography is better than Unsplash:**
With Unsplash, you only have the downloaded JPEG as the "original" —
the camera RAW is unavailable. With your own photos, you can compare
Instagram's output against the uncompressed camera file, not against
a previously-compressed JPEG. This removes one source of noise from
the measurement.

---

### 3.6 Additional Quality Metrics

**LPIPS (Learned Perceptual Image Patch Similarity)**

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).
The unreasonable effectiveness of deep features as a perceptual metric.
CVPR 2018. arXiv:1801.03924

Installation: pip install lpips (requires PyTorch)

LPIPS uses deep CNN feature distances rather than pixel-level comparison.
Shown to correlate better with human perceptual judgments than SSIM on
most distortion types.

Key question: does LPIPS show the same category interaction pattern as SSIM?
If architecture degrades most and gradient least on LPIPS as well, the
result is robust across metric families. If the ordering differs, that
is itself a finding worth reporting.

**Color-space SSIM**

Current SSIM implementation converts to grayscale, ignoring color distortion.
Instagram's pipeline may shift colors (saturation, white balance) in ways
that are perceptible but invisible to grayscale SSIM.

Implementation: compute SSIM separately on each RGB channel and report
the channel-wise results alongside the grayscale result.

---

### 3.7 Statistical Assumption Checking

Current analysis reports mixed ANOVA results without formally checking
the assumptions the model requires. A peer reviewer will ask for these.

**Checks to add to mixed_model.py:**

1. Mauchly's test of sphericity
   Tests whether the variance of the differences between all pairs of
   within-subject conditions is equal. Required for repeated-measures ANOVA.
   pingouin.sphericity() implements this directly.

2. Shapiro-Wilk test of normality on residuals
   Tests whether model residuals are normally distributed.
   scipy.stats.shapiro() on the residuals from the model.

3. Levene's test of homogeneity of variance
   Tests whether variance is equal across groups.
   scipy.stats.levene() across category groups.

4. Omega-squared effect size
   Less biased than partial eta-squared for small samples.
   Formula: omega2 = (SS_effect - df_effect * MS_error) /
                     (SS_total + MS_error)
   Report alongside eta-squared for completeness.

---

## v4: Machine Learning Pipeline

### 4.1 Instagram Compression Simulator

**Goal:** Learn a function F such that F(x) approximates Instagram(x)
for any input image x.

**Training data:** The empirical dataset from v3 (upload/download pairs).
Target size: N >= 500 image pairs covering all 8 categories and 5 aspect
ratios, including both single posts and carousel positions 1 and 2.

**Architecture:** Convolutional autoencoder.
- Encoder: 4-5 convolutional layers, stride 2 downsampling
- Bottleneck: learned representation of the compression state
- Decoder: transposed convolutions, skip connections (U-Net style)
- Loss: LPIPS + L1 pixel loss + adversarial loss (optional)

**Why adversarial loss:** A pure LPIPS + L1 loss will produce blurry
outputs that minimize average error. An adversarial component (a discriminator
trained to distinguish real Instagram outputs from simulated ones) pushes
the generator to produce sharp, realistic compression artifacts.

**Evaluation:** SSIM, PSNR, LPIPS between F(x) and actual Instagram(x)
on a held-out test set. Report separately by category and carousel position.

**Key question for the carousel model:** Does Instagram apply a single
shared compression to all images in a carousel, or does each image get
compressed independently? If independently, F is the same function applied
to each image. If shared, F takes the entire sequence as input.

---

### 4.2 Preprocessing Optimizer

**Goal:** Given a learned differentiable model F of Instagram's pipeline,
find the preprocessing transformation T that minimizes quality loss:

    T* = argmin_{T in T_set} LPIPS(original, F(T(original)))

where T_set is the space of valid preprocessing transformations
(resize method, padding color, seam carving energy function parameters).

**Optimization approach:**
If T is parameterized continuously (e.g., padding color as RGB values,
seam carving weights as alpha/beta/gamma), gradient descent through F
finds the optimal parameters directly.

If T is discrete (e.g., choose between simple resize, padding, seam
carving), use a lookup table or train a classifier that maps image
content features to optimal T.

**Expected output:** A function that takes any image and returns the
preprocessing strategy that minimizes post-Instagram quality loss.
Deployable as a command-line tool or web service.

---

### 4.3 Carousel Sequence Model

**Goal:** Model the joint compression of a sequence of carousel images,
accounting for the aspect ratio enforcement from image 1.

**Key unknowns to resolve empirically:**
- Does Instagram resize all images to match image 1's aspect ratio before
  or after JPEG compression?
- Does the resize algorithm differ between image 1 and subsequent images?
- Is there a quality difference between carousel position 1 and position 2+?

**Model structure:**
- Step 1: Aspect ratio enforcement (deterministic, learned from data)
- Step 2: JPEG compression (learned per-position compression model)
- Step 3: Multi-resolution storage and serving (currently out of scope)

---

## v5: Computational Textbook

### Structure

A Jupyter Book (or folder of .ipynb notebooks) that teaches the mathematics
and implementation of every concept in this project, from first principles
to the full ML pipeline. Each chapter alternates between mathematical proofs
and Python code that validates or demonstrates the math.

Inspired by Casella & Berger's Statistical Inference for rigor, but
visually modern and computationally interactive. Target audience: someone
with undergraduate math (calculus, linear algebra, probability) who wants
to understand not just how to use these tools but why they work.

**Proposed chapters:**

Part I: Images as Mathematical Objects
- Chapter 1: The image as a function (arrays, color spaces, pixel distributions)
- Chapter 2: Measuring image difference (MSE, PSNR, SSIM — full derivations)
- Chapter 3: Convolution and image filtering (Gaussian blur, gradient operators)

Part II: Resize Methods
- Chapter 4: Geometric transformations and interpolation theory
- Chapter 5: Aspect ratio arithmetic and the padding cascade (algebraic proof
  of the double-resize degradation)
- Chapter 6: Seam carving (Bellman equation, DP proof, energy functions,
  forward vs backward energy)

Part III: The Instagram Pipeline
- Chapter 7: JPEG compression (DCT, quantization tables, quality factor math)
- Chapter 8: Compounding compression (why double-JPEG is worse than single-JPEG,
  formal derivation)

Part IV: Statistical Analysis
- Chapter 9: Factorial experimental design (why crossing factors matters)
- Chapter 10: Mixed ANOVA derivation (F-statistic, SS decomposition, random effects)
- Chapter 11: Effect sizes (Cohen's d, eta-squared, omega-squared — derivations
  and interpretations)

Part V: The ML Approach
- Chapter 12: Learning a compression simulator (convolutional autoencoders,
  LPIPS loss, adversarial training)
- Chapter 13: Preprocessing optimization (gradient descent through a learned model)

**Format:** Each chapter is a single .ipynb notebook. Math in markdown cells
with LaTeX. Code in Python cells with outputs. Visualizations use matplotlib
with the project palette.

**Starting point:** Chapter 2 (SSIM derivation) — self-contained, validates
our existing implementation, and is the most commonly misunderstood metric
in the project.

---

## Publication Plan

**Target 1: LinkedIn article (short term)**
Lead with the counterintuitive finding. Include the per-image scatter plot
colored by category and one before/after visual from test_single_image.py
on an architecture photo. ~800 words. Requires: one real architectural photo
run through the pipeline before we can publish.

**Target 2: Technical blog post on portfolio site (short term)**
Deeper methodology write-up. Include the full category table, ANOVA results,
and mechanism explanation. Links to the repo. ~2000 words.

**Target 3: arXiv preprint (medium term, after v3)**
Full paper: introduction, related work, methods, results, discussion.
Target: cs.CV (computer vision). Requires: real image results,
empirical pipeline verification, assumption checks.

**Target 4: Conference or journal (long term, after v4)**
Once the ML optimizer exists, the full arc from problem identification to
learned solution is a complete conference paper.
Possible venues: ICIP (Image and Communication Processing),
CVPR workshop, or Journal of Image and Video Processing.

---

## Open Questions

These are unresolved questions that future work should address:

1. Does Instagram's compression differ meaningfully between mobile and
   desktop upload? (Empirical verification will answer this.)

2. Does the "High Quality Uploads" setting actually change Instagram's
   compression, or is it marketing? (Testable: upload same image both ways.)

3. Is Instagram's compression deterministic? Upload the same image twice —
   are the outputs byte-for-byte identical?

4. Does LPIPS show the same category ordering as SSIM? (Will be answered
   in v3.)

5. Is there a preprocessing strategy that outperforms simple resize for
   any category? (Seam carving might win on portrait-oriented content with
   saliency energy. Will be answered in v3.)

6. Does the double-resize degradation scale linearly with the amount of
   padding, or is there a threshold effect? (Aspect ratio interaction
   result suggests linear, but worth formal testing.)

7. For carousel posts, does the optimal preprocessing strategy for image 2+
   depend on what image 1 is? (The sequence model in v4 will answer this.)

---

*Last updated: March 2026*
*Project status: Paused for field data collection. See PHOTO_PROTOCOL.md.*
