# src/metrics.py
#
# Image quality metrics for IGPhotoResizer.
# All functions operate on PIL Images and return scalar float values.
# Metrics are always computed over the content region only — never over padding.

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as skimage_ssim


# ---------------------------------------------------------------------------
# Region of Interest extraction
# ---------------------------------------------------------------------------

def get_content_region(
    original: Image.Image,
    resized_result,  # ResizeResult or PIL Image
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the comparable content region from both images for metric computation.

    If resized_result has a content_box (padding was added), we crop that region
    out of the processed image and compare it directly against the original —
    no round-trip resize needed, no border bleed.

    If content_box is None (simple resize or seam carving), we resize the
    processed image back to original dimensions using LANCZOS for comparison.

    This is the correct fix for the metric bias identified in the first
    experiment run, where padding resize appeared to score dramatically worse
    than simple resize purely because border pixels were contaminating the score.
    """
    from src.methods import ResizeResult

    if isinstance(resized_result, ResizeResult):
        processed_image = resized_result.image
        content_box = resized_result.content_box
    else:
        # backwards compatibility: accept plain PIL Image
        processed_image = resized_result
        content_box = None

    if content_box is not None:
        # Crop content region and resize to original dimensions for comparison
        x0, y0, x1, y1 = content_box
        content_crop = processed_image.crop(content_box)
        content_matched = content_crop.resize(original.size, Image.Resampling.LANCZOS)
    else:
        # No padding — round-trip resize back to original dimensions
        content_matched = processed_image.resize(original.size, Image.Resampling.LANCZOS)

    original_array = np.array(original, dtype=np.float64)
    resized_array  = np.array(content_matched, dtype=np.float64)

    return original_array, resized_array


def to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB float array to grayscale using luminance weights.

    The standard luminance formula (ITU-R BT.601) is:
        Y = 0.299 * R + 0.587 * G + 0.114 * B

    These weights reflect human perceptual sensitivity — we are most sensitive
    to green, less to red, least to blue. This is why grayscale conversion
    is not a simple mean across channels.

    Compared to the legacy code which used PIL's ImageOps.grayscale (which
    applies the same formula internally), this version operates directly on
    numpy arrays so we can control it explicitly.
    """
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image_array, weights)


# ---------------------------------------------------------------------------
# Metric 1: MSE
# ---------------------------------------------------------------------------

def compute_mse(
    original: Image.Image,
    resized: Image.Image
) -> float:
    """
    Mean Squared Error between original and resized image.

    MSE = (1 / N) * sum((A - B)^2)

    where N is the total number of pixels * channels.

    Computed in RGB space (all three channels), not grayscale, so color
    shifts are captured. Range is [0, 65025] for uint8 images (255^2).
    Lower is better.

    Known limitation: MSE is perceptually non-uniform. A uniform +10 shift
    in brightness produces the same MSE as random +10 noise, but the former
    is nearly invisible to humans and the latter is very visible.
    """
    original_array, resized_array = get_content_region(original, resized)
    mse = np.mean((original_array - resized_array) ** 2)
    print(f"    MSE: {mse:.4f}")
    return float(mse)


# ---------------------------------------------------------------------------
# Metric 2: PSNR
# ---------------------------------------------------------------------------

def compute_psnr(
    original: Image.Image,
    resized: Image.Image,
    max_pixel_value: float = 255.0
) -> float:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(MAX^2 / MSE)

    where MAX is the maximum possible pixel value (255 for uint8).

    PSNR is expressed in decibels (dB) and is more interpretable than raw MSE:
        > 40 dB  : excellent, virtually indistinguishable from original
        30-40 dB : good quality, minor artifacts
        20-30 dB : acceptable, visible degradation
        < 20 dB  : poor quality

    PSNR is just a log-scaled MSE — it has the same perceptual limitations,
    but the dB scale is more intuitive for reporting.

    Returns float('inf') if MSE is 0 (identical images).
    """
    original_array, resized_array = get_content_region(original, resized)
    mse = np.mean((original_array - resized_array) ** 2)

    if mse == 0:
        print(f"    PSNR: inf (identical images)")
        return float('inf')

    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    print(f"    PSNR: {psnr:.4f} dB")
    return float(psnr)


# ---------------------------------------------------------------------------
# Metric 3: SSIM
# ---------------------------------------------------------------------------

def compute_ssim(
    original: Image.Image,
    resized: Image.Image
) -> float:
    """
    Structural Similarity Index (SSIM).

    SSIM(x, y) = [l(x,y)]^alpha * [c(x,y)]^beta * [s(x,y)]^gamma

    With alpha=beta=gamma=1:
        l(x,y) = (2*mu_x*mu_y + C1) / (mu_x^2 + mu_y^2 + C1)        # luminance
        c(x,y) = (2*sigma_x*sigma_y + C2) / (sigma_x^2 + sigma_y^2 + C2)  # contrast
        s(x,y) = (sigma_xy + C3) / (sigma_x*sigma_y + C3)             # structure

    where mu = local mean, sigma = local std dev, sigma_xy = local covariance,
    computed over an 11x11 Gaussian-weighted window sliding across the image.

    The structure term s(x,y) is essentially Pearson's r between the two
    local patches — it asks whether the pattern of variation is preserved,
    independent of brightness or contrast differences.

    Computed on grayscale (luminance) to match the original SSIM paper.
    Range is [-1, 1], higher is better. In practice values are typically
    between 0.7 and 1.0 for reasonable resize operations.

    We use skimage's implementation which applies the Gaussian window
    via convolution (the operation we'll formalize when we cover convolutions).
    """
    original_array, resized_array = get_content_region(original, resized)

    original_gray = to_grayscale(original_array)
    resized_gray  = to_grayscale(resized_array)

    # data_range is the range of the input data — 255 for uint8-equivalent float
    ssim_value = skimage_ssim(
        original_gray,
        resized_gray,
        data_range=255.0,
        win_size=11,
        gaussian_weights=True
    )
    print(f"    SSIM: {ssim_value:.6f}")
    return float(ssim_value)


# ---------------------------------------------------------------------------
# Combined metric computation
# ---------------------------------------------------------------------------
def compute_all_metrics(
    original: Image.Image,
    resized_result,  # ResizeResult or PIL Image
    method_name: str,
    image_name: str
) -> dict:
    print(f"  Computing metrics for [{method_name}] on [{image_name}]")
    mse  = compute_mse(original, resized_result)
    psnr = compute_psnr(original, resized_result)
    ssim = compute_ssim(original, resized_result)

    return {
        "image":       image_name,
        "method":      method_name,
        "mse":         mse,
        "psnr_db":     psnr,
        "ssim":        ssim,
        "orig_width":  original.size[0],
        "orig_height": original.size[1],
    }