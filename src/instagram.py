# src/instagram.py
#
# Simulates Instagram's image processing pipeline.
#
# When you upload an image to Instagram, it applies two transformations:
#   1. Resizes to a fixed target dimension based on post type
#   2. JPEG compresses at roughly quality 70-85
#
# This means your preprocessing choice doesn't just affect the image you
# upload — it affects the image that survives Instagram's pipeline. A method
# that looks good before upload might degrade more or less than another method
# after Instagram processes it. That's the core experimental question.
#
# Instagram's exact pipeline is not public. These parameters are based on
# community reverse-engineering and are reasonable approximations.

import io
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Instagram dimension targets (as of 2024, best available approximations)
# ---------------------------------------------------------------------------

INSTAGRAM_TARGETS = {
    "square":    (1080, 1080),   # 1:1 aspect ratio
    "portrait":  (1080, 1350),   # 4:5 aspect ratio (most common for feed)
    "landscape": (1080,  566),   # 1.91:1 aspect ratio
}

# Instagram's approximate JPEG compression quality
INSTAGRAM_JPEG_QUALITY = 75


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def jpeg_compress(image: Image.Image, quality: int = INSTAGRAM_JPEG_QUALITY) -> Image.Image:
    """
    Simulate JPEG compression at a given quality level.

    JPEG compression works in several stages:
      1. Convert RGB to YCbCr color space (separates luminance from color)
      2. Downsample the color channels (humans are less sensitive to color
         resolution than brightness resolution)
      3. Apply the Discrete Cosine Transform (DCT) to 8x8 pixel blocks,
         converting spatial information to frequency information
      4. Quantize the DCT coefficients — this is where information is
         permanently lost. Low-frequency components (broad shapes) are
         kept more precisely; high-frequency components (fine detail,
         sharp edges) are quantized more aggressively.
      5. Entropy encode (Huffman coding) the quantized coefficients

    The quality parameter (1-95) controls the quantization step:
    higher quality = finer quantization = less loss = larger file.
    Quality 75 is Instagram's approximate setting — enough to be
    visibly lossy on close inspection but acceptable at normal viewing.

    We simulate this by encoding to a bytes buffer and decoding back.
    The round-trip through JPEG encoding/decoding applies all the above
    steps and gives us the actual degraded pixel values.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed.load()  # force decode before buffer goes out of scope
    print(f"    JPEG compressed at quality={quality}: "
          f"original ~{image.size[0]*image.size[1]*3/1024:.0f}KB -> "
          f"compressed ~{buffer.tell()/1024:.0f}KB")
    return compressed


def instagram_resize(
    image: Image.Image,
    post_type: str = "portrait"
) -> Image.Image:
    """
    Resize image to Instagram's target dimensions for the given post type.

    Instagram resizes using a high-quality filter (believed to be bicubic
    or Lanczos). We use LANCZOS here as the best available approximation.

    The post_type determines the target dimensions:
        square:    1080x1080  (1:1)
        portrait:  1080x1350  (4:5)  <- most common for feed posts
        landscape: 1080x566   (1.91:1)

    Note: Instagram fits the image within these dimensions while preserving
    aspect ratio — it does not squash. Any remaining space becomes a border
    in the app's display, but the image file itself is cropped to the target.
    For our simulation we resize to fit within the target.
    """
    if post_type not in INSTAGRAM_TARGETS:
        raise ValueError(f"Unknown post_type '{post_type}'. "
                         f"Choose from: {list(INSTAGRAM_TARGETS.keys())}")

    target_width, target_height = INSTAGRAM_TARGETS[post_type]

    # Fit within target while preserving aspect ratio
    original_width, original_height = image.size
    scale = min(target_width / original_width, target_height / original_height)
    new_width  = int(original_width  * scale)
    new_height = int(original_height * scale)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"    Instagram resize ({post_type}): {image.size} -> {resized.size}")
    return resized


def run_instagram_pipeline(
    image: Image.Image,
    post_type: str = "portrait",
    jpeg_quality: int = INSTAGRAM_JPEG_QUALITY
) -> Image.Image:
    """
    Apply the full simulated Instagram pipeline to an image:
        Step 1: Resize to Instagram target dimensions
        Step 2: JPEG compress at Instagram quality

    This represents what your image looks like after Instagram processes it.
    The key insight: we apply this pipeline AFTER our preprocessing step,
    so the full experiment flow is:

        original -> [our resize method] -> [instagram pipeline] -> measure quality

    This lets us ask: which preprocessing method produces the best final
    image quality after surviving Instagram's compression?
    """
    print(f"  Running Instagram pipeline (post_type={post_type}, "
          f"jpeg_quality={jpeg_quality})")
    step1 = instagram_resize(image, post_type)
    step2 = jpeg_compress(step1, jpeg_quality)
    return step2


# ---------------------------------------------------------------------------
# Pipeline variants for sensitivity analysis
# ---------------------------------------------------------------------------

PIPELINE_VARIANTS = {
    "instagram_standard": lambda img: run_instagram_pipeline(img, "portrait", 75),
    "instagram_high_quality": lambda img: run_instagram_pipeline(img, "portrait", 85),
    "instagram_low_quality": lambda img: run_instagram_pipeline(img, "portrait", 70),
    "instagram_square": lambda img: run_instagram_pipeline(img, "square", 75),
    "instagram_landscape": lambda img: run_instagram_pipeline(img, "landscape", 75),
}