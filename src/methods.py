# src/methods.py
#
# Canonical resize method implementations for IGPhotoResizer.
# Each function takes a PIL Image and target dimensions, and returns a ResizeResult.
# No file I/O here — callers handle loading and saving.

import numpy as np
from PIL import Image
from colorthief import ColorThief
from dataclasses import dataclass
import io


# ---------------------------------------------------------------------------
# ResizeResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResizeResult:
    """
    Return type for all resize methods.

    image:       The resized PIL Image at target dimensions
    content_box: (x0, y0, x1, y1) pixel coordinates of the original content
                 within the output image. None means the entire image is content
                 (no padding was added). Used by metrics.py to evaluate only
                 the content region, ignoring any border fill.
    method:      Name of the resize method that produced this result
    metadata:    Optional dict for method-specific diagnostics (scale factor,
                 seams removed, dominant color, etc.)

    Future border methods (edge blur, outpainting) return the same structure —
    content_box always marks where the original content lives regardless of
    what fills the border. This means metrics.py never needs to change.
    """
    image:       Image.Image
    content_box: tuple | None  # (x0, y0, x1, y1) or None
    method:      str
    metadata:    dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


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
) -> ResizeResult:
    """
    Resize image directly to (target_width, target_height) using LANCZOS resampling.

    Does NOT preserve aspect ratio — a 9:16 portrait photo becomes a squashed
    square. This distortion is intentional: it represents the worst-case naive
    resize and gives us a baseline to compare against.

    content_box is None because the entire output image is content — there is
    no padding, just distortion.
    """
    print(f"  [simple_resize] {image.size} -> ({target_width}x{target_height})")
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return ResizeResult(
        image=resized,
        content_box=None,
        method="simple_resize",
        metadata={"target_width": target_width, "target_height": target_height}
    )


# ---------------------------------------------------------------------------
# Method 2: Padding Resize (aspect-ratio preserving + dominant color fill)
# ---------------------------------------------------------------------------

def padding_resize(
    image: Image.Image,
    target_width: int = 1080,
    target_height: int = 1080
) -> ResizeResult:
    """
    Resize image to fit within target dimensions preserving aspect ratio,
    then fill remaining space with the image's dominant color.

    Returns content_box=(x0, y0, x1, y1) marking exactly where the original
    content was pasted on the canvas. Metrics are evaluated only within this
    box so border pixels never contaminate the quality score.

    Border fill roadmap:
        v1 (current): dominant color flat fill
        v2:           edge-blurred / mirrored fill
        v3:           outpainting via generative model
    All future variants return the same content_box structure.
    """
    print(f"  [padding_resize] {image.size} -> ({target_width}x{target_height})")

    original_width, original_height = image.size
    scaled_width, scaled_height = compute_scale_factor(
        original_width, original_height, target_width, target_height
    )

    resized_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

    dominant_color = get_dominant_color_from_image(image)
    canvas = Image.new("RGB", (target_width, target_height), dominant_color)

    x_offset = (target_width - scaled_width) // 2
    y_offset = (target_height - scaled_height) // 2
    canvas.paste(resized_image, (x_offset, y_offset))
    print(f"    Pasted at offset ({x_offset}, {y_offset})")

    content_box = (
        x_offset,
        y_offset,
        x_offset + scaled_width,
        y_offset + scaled_height
    )

    return ResizeResult(
        image=canvas,
        content_box=content_box,
        method="padding_resize",
        metadata={
            "dominant_color": dominant_color,
            "scaled_width":   scaled_width,
            "scaled_height":  scaled_height,
            "x_offset":       x_offset,
            "y_offset":       y_offset,
        }
    )


# ---------------------------------------------------------------------------
# Method 3: Seam Carving (true content-aware resize)
# ---------------------------------------------------------------------------

def compute_energy_map(image_array: np.ndarray) -> np.ndarray:
    """
    Compute the energy map of an image using gradient magnitude.

    Energy measures how much a pixel differs from its neighbors.
    High energy = edges and texture. Low energy = uniform regions.
    Seam carving removes low-energy paths to minimize perceptual damage.

    L1 gradient:
        energy(i,j) = |∂I/∂x| + |∂I/∂y|

    Partial derivatives approximated by finite differences:
        ∂I/∂x ≈ I(i, j+1) - I(i, j-1)
        ∂I/∂y ≈ I(i+1, j) - I(i-1, j)
    """
    gray = np.mean(image_array, axis=2)
    grad_x = np.abs(np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1))
    grad_y = np.abs(np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
    energy = grad_x + grad_y
    print(f"    Energy map: shape={energy.shape}, mean={energy.mean():.2f}, max={energy.max():.2f}")
    return energy


def find_minimum_energy_seam(energy_map: np.ndarray) -> np.ndarray:
    """
    Find the vertical seam with minimum total energy using dynamic programming.

    DP recurrence:
        M(i,j) = energy(i,j) + min(M(i-1,j-1), M(i-1,j), M(i-1,j+1))

    Backtrack from minimum in last row to recover the seam path.
    O(H * W) time.
    """
    height, width = energy_map.shape
    cumulative_energy = energy_map.copy()

    for row in range(1, height):
        for col in range(width):
            left   = cumulative_energy[row - 1, max(col - 1, 0)]
            center = cumulative_energy[row - 1, col]
            right  = cumulative_energy[row - 1, min(col + 1, width - 1)]
            cumulative_energy[row, col] += min(left, center, right)

    seam = np.zeros(height, dtype=int)
    seam[-1] = np.argmin(cumulative_energy[-1])

    for row in range(height - 2, -1, -1):
        prev_col = seam[row + 1]
        left   = cumulative_energy[row, max(prev_col - 1, 0)]
        center = cumulative_energy[row, prev_col]
        right  = cumulative_energy[row, min(prev_col + 1, width - 1)]
        offset = np.argmin([left, center, right]) - 1
        seam[row] = np.clip(prev_col + offset, 0, width - 1)

    return seam


def remove_seam(image_array: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """
    Remove a vertical seam from an image array.
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
) -> ResizeResult:
    """
    Resize image to target_width by iteratively removing lowest-energy vertical seams.

    Seam carving never adds padding — it only removes content. So content_box
    is None (the entire output is content, just with some columns removed).

    Pre-scales height to target_height first via LANCZOS, then removes seams
    to reach target_width.
    """
    print(f"  [seam_carving_resize] {image.size} -> ({target_width}x{target_height})")

    original_width, original_height = image.size

    if original_height != target_height:
        scale = target_height / original_height
        prescaled_width = int(original_width * scale)
        image = image.resize((prescaled_width, target_height), Image.Resampling.LANCZOS)
        print(f"    Pre-scaled to ({prescaled_width}x{target_height})")

    current_width = image.size[0]
    seams_to_remove = current_width - target_width

    if seams_to_remove < 0:
        print(f"    WARNING: target_width ({target_width}) > current width ({current_width}). Skipping.")
        return ResizeResult(
            image=image,
            content_box=None,
            method="seam_carving_resize",
            metadata={"seams_removed": 0, "warning": "target wider than source"}
        )

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

    return ResizeResult(
        image=result,
        content_box=None,
        method="seam_carving_resize",
        metadata={"seams_removed": seams_to_remove}
    )


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "simple_resize":       simple_resize,
    "padding_resize":      padding_resize,
    "seam_carving_resize": seam_carving_resize,
}
