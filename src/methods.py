# src/methods.py
#
# Canonical resize method implementations for IGPhotoResizer.
# Each function takes a PIL Image and target dimensions, and returns a PIL Image.
# No file I/O here — callers handle loading and saving.

import numpy as np
from PIL import Image
from colorthief import ColorThief
import io


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_dominant_color_from_image(pil_image: Image.Image) -> tuple[int, int, int]:
    """
    Extract the dominant color from a PIL Image using ColorThief.
    ColorThief requires a file path or file-like object, so we serialize
    the image to a bytes buffer first.
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    dominant_color = ColorThief(buffer).get_color(quality=1)
    print(f"    Dominant color extracted: RGB{dominant_color}")
    return dominant_color


def compute_scale_factor(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int
) -> tuple[int, int]:
    """
    Compute the largest dimensions that fit within (target_width, target_height)
    while preserving the original aspect ratio.

    This is proportional scaling — the core math is:
        scale = min(target_width / original_width, target_height / original_height)
        new_width  = floor(original_width  * scale)
        new_height = floor(original_height * scale)

    Using min() ensures neither dimension exceeds its target.
    """
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    print(f"    Scale factor: {scale:.4f} -> ({original_width}x{original_height}) => ({new_width}x{new_height})")
    return new_width, new_height


# ---------------------------------------------------------------------------
# Method 1: Simple Resize (LANCZOS)
# ---------------------------------------------------------------------------

def simple_resize(
    image: Image.Image,
    target_width: int = 1080,
    target_height: int = 1080
) -> Image.Image:
    """
    Resize image directly to (target_width, target_height) using LANCZOS resampling.

    LANCZOS (also called Sinc in signal processing) works by convolving the image
    with a windowed sinc kernel. It is the highest quality resampling filter in
    Pillow for downscaling — it considers a neighborhood of pixels rather than
    just the nearest one or two, which reduces aliasing artifacts.

    NOTE: This does NOT preserve aspect ratio. A 9:16 portrait photo resized to
    1080x1080 will be squashed into a square. This distortion is intentional here
    so we can measure it — it represents the worst case for naive resizing.
    """
    print(f"  [simple_resize] {image.size} -> ({target_width}x{target_height})")
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return resized


# ---------------------------------------------------------------------------
# Method 2: Padding Resize (aspect-ratio preserving + dominant color fill)
# ---------------------------------------------------------------------------

def padding_resize(
    image: Image.Image,
    target_width: int = 1080,
    target_height: int = 1080
) -> Image.Image:
    """
    Resize image to fit within (target_width, target_height) while preserving
    aspect ratio, then fill remaining space with the image's dominant color.

    Steps:
        1. Compute scaled dimensions that fit within target while preserving ratio
        2. Resize to those dimensions using LANCZOS
        3. Create a blank canvas of (target_width, target_height) filled with
           the dominant color of the original image
        4. Paste the resized image centered on the canvas

    The dominant color fill is the key improvement over naive white padding —
    it makes the borders visually blend with the image content, which looks
    more natural on Instagram.
    """
    print(f"  [padding_resize] {image.size} -> ({target_width}x{target_height})")

    original_width, original_height = image.size
    scaled_width, scaled_height = compute_scale_factor(
        original_width, original_height, target_width, target_height
    )

    # Step 1+2: proportional resize
    resized_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

    # Step 3: canvas filled with dominant color
    dominant_color = get_dominant_color_from_image(image)
    canvas = Image.new("RGB", (target_width, target_height), dominant_color)

    # Step 4: center the resized image on the canvas
    x_offset = (target_width - scaled_width) // 2
    y_offset = (target_height - scaled_height) // 2
    canvas.paste(resized_image, (x_offset, y_offset))
    print(f"    Pasted at offset ({x_offset}, {y_offset})")

    return canvas


# ---------------------------------------------------------------------------
# Method 3: Seam Carving (true content-aware resize)
# ---------------------------------------------------------------------------

def compute_energy_map(image_array: np.ndarray) -> np.ndarray:
    """
    Compute the energy map of an image using gradient magnitude.

    Energy measures how much a pixel differs from its neighbors — pixels on
    edges or in areas of high contrast have high energy, uniform regions have
    low energy. Seam carving removes low-energy paths because removing them
    causes the least perceptual damage.

    We use the L1 gradient (sum of absolute differences in x and y directions):
        energy(i,j) = |∂I/∂x| + |∂I/∂y|

    where the partial derivatives are approximated by finite differences:
        ∂I/∂x at (i,j) ≈ I(i, j+1) - I(i, j-1)   (horizontal neighbor diff)
        ∂I/∂y at (i,j) ≈ I(i+1, j) - I(i-1, j)   (vertical neighbor diff)

    We compute this on a grayscale version of the image (luminance channel),
    then use np.roll for efficient vectorized neighbor access.
    """
    # Convert to grayscale luminance for energy computation
    gray = np.mean(image_array, axis=2)  # shape: (H, W)

    # Finite difference gradients using np.roll (wraps edges, acceptable for seam carving)
    grad_x = np.abs(np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1))
    grad_y = np.abs(np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))

    energy = grad_x + grad_y
    print(f"    Energy map computed: shape={energy.shape}, mean={energy.mean():.2f}, max={energy.max():.2f}")
    return energy


def find_minimum_energy_seam(energy_map: np.ndarray) -> np.ndarray:
    """
    Find the vertical seam (top-to-bottom path) with minimum total energy
    using dynamic programming.

    A vertical seam is a connected path of pixels, one per row, where each
    pixel is horizontally adjacent (±1 column) to the pixel in the row above.

    The DP recurrence is:
        M(i, j) = energy(i, j) + min(M(i-1, j-1), M(i-1, j), M(i-1, j+1))

    where M(i,j) is the minimum cumulative energy to reach row i at column j.
    The seam is found by backtracking from the minimum value in the last row.

    This is O(H * W) time — optimal for this problem.
    """
    height, width = energy_map.shape
    cumulative_energy = energy_map.copy()

    # Forward pass: fill cumulative energy table row by row
    for row in range(1, height):
        for col in range(width):
            left   = cumulative_energy[row - 1, max(col - 1, 0)]
            center = cumulative_energy[row - 1, col]
            right  = cumulative_energy[row - 1, min(col + 1, width - 1)]
            cumulative_energy[row, col] += min(left, center, right)

    # Backtrack from the minimum in the last row
    seam = np.zeros(height, dtype=int)
    seam[-1] = np.argmin(cumulative_energy[-1])

    for row in range(height - 2, -1, -1):
        prev_col = seam[row + 1]
        left   = cumulative_energy[row, max(prev_col - 1, 0)]
        center = cumulative_energy[row, prev_col]
        right  = cumulative_energy[row, min(prev_col + 1, width - 1)]
        offset = np.argmin([left, center, right]) - 1
        seam[row] = np.clip(prev_col + offset, 0, width - 1)

    print(f"    Seam found: min cumulative energy = {cumulative_energy[-1, seam[-1]]:.2f}")
    return seam


def remove_seam(image_array: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """
    Remove a vertical seam from an image array.

    For each row i, delete the pixel at column seam[i].
    Result has shape (H, W-1, C).
    """
    height, width, channels = image_array.shape
    output = np.zeros((height, width - 1, channels), dtype=image_array.dtype)

    for row in range(height):
        col = seam[row]
        output[row, :, :] = np.delete(image_array[row, :, :], col, axis=0)

    return output


def seam_carving_resize(
    image: Image.Image,
    target_width: int = 1080,
    target_height: int = 1080
) -> Image.Image:
    """
    Resize image to target_width by iteratively removing lowest-energy vertical seams.

    This only removes columns (reduces width). If the image also needs height
    reduction, we first apply proportional scaling to get close to the target
    height, then use seam carving to reach the exact target width.

    WARNING: Seam carving is O(H * W) per seam, and we may need to remove
    hundreds of seams. This is intentionally slow for now — we will optimize
    with vectorized DP in a later version.
    """
    print(f"  [seam_carving_resize] {image.size} -> ({target_width}x{target_height})")

    original_width, original_height = image.size

    # Step 1: proportionally scale height to target_height first if needed
    if original_height != target_height:
        scale = target_height / original_height
        prescaled_width = int(original_width * scale)
        image = image.resize((prescaled_width, target_height), Image.Resampling.LANCZOS)
        print(f"    Pre-scaled to ({prescaled_width}x{target_height})")

    current_width = image.size[0]
    seams_to_remove = current_width - target_width

    if seams_to_remove < 0:
        print(f"    WARNING: target_width ({target_width}) > current width ({current_width}). Skipping seam carving.")
        return image

    print(f"    Removing {seams_to_remove} seams...")
    image_array = np.array(image, dtype=np.float64)

    for i in range(seams_to_remove):
        if i % 50 == 0:
            print(f"    Seam {i}/{seams_to_remove}")
        energy_map = compute_energy_map(image_array)
        seam = find_minimum_energy_seam(energy_map)
        image_array = remove_seam(image_array, seam)

    result = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
    print(f"    Seam carving complete: final size = {result.size}")
    return result


# ---------------------------------------------------------------------------
# Method registry — used by experiments/run_experiment.py
# ---------------------------------------------------------------------------

METHODS = {
    "simple_resize":      simple_resize,
    "padding_resize":     padding_resize,
    "seam_carving_resize": seam_carving_resize,
}