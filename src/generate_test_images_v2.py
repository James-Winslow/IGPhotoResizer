# src/generate_test_images_v2.py
#
# Generates the synthetic_v2 test set for the fully crossed factorial experiment.
#
# Design: 8 categories × 5 aspect ratios × 3 replicates = 120 images
#
# Each image is named: {category}_{aspect_ratio}_r{replicate}.png
# Example: nebula_portrait_r2.png
#
# This naming convention allows run_experiment.py to extract category and
# aspect_ratio as experimental factors automatically, enabling the full
# mixed model analysis.
#
# Usage:
#   python src/generate_test_images_v2.py
#   python src/generate_test_images_v2.py --output-dir test_sets/synthetic_v2

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_sets", "synthetic_v2")

# ---------------------------------------------------------------------------
# Aspect ratio definitions
# Each tuple is (name, width, height)
# Heights are anchored to 1080 for consistency with Instagram target
# ---------------------------------------------------------------------------

ASPECT_RATIOS = [
    ("very_wide",  1920, 960),   # 2:1
    ("wide",       1440, 1080),  # 4:3
    ("square",     1080, 1080),  # 1:1
    ("portrait",    810, 1080),  # 3:4
    ("very_tall",   540, 1080),  # 1:2
]

REPLICATES = 3

# Base seeds — each replicate gets a different seed derived from this
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Utility: smooth noise using sum of sine waves
# Produces organic-looking gradients without requiring external noise libs
# ---------------------------------------------------------------------------

def sine_noise(width: int, height: int, frequency: float, angle: float,
               rng: np.random.Generator) -> np.ndarray:
    """Single sine wave across the image at a given frequency and angle."""
    x = np.linspace(0, frequency * 2 * np.pi, width)
    y = np.linspace(0, frequency * 2 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    phase = rng.uniform(0, 2 * np.pi)
    return np.sin(xx * np.cos(angle) + yy * np.sin(angle) + phase)


def layered_noise(width: int, height: int, n_layers: int,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Sum of sine waves at different frequencies and angles.
    Produces smooth organic texture — the basis for most category generators.
    This is a simplified version of Perlin noise using Fourier composition.
    Each layer doubles the frequency (octave doubling) and halves the amplitude,
    which is how natural textures are structured fractally.
    """
    result = np.zeros((height, width))
    amplitude = 1.0
    frequency = 1.0
    for _ in range(n_layers):
        angle = rng.uniform(0, np.pi)
        result += amplitude * sine_noise(width, height, frequency, angle, rng)
        amplitude *= 0.5
        frequency *= 2.0
    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


# ---------------------------------------------------------------------------
# Category generators
# Each function takes (width, height, rng) and returns a PIL Image
# ---------------------------------------------------------------------------

def generate_nebula(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """
    Deep space nebula: smooth color gradients with star points.
    Characteristic: large-scale smooth variation, occasional bright points,
    deep color palette (purples, blues, magentas).
    """
    # Base nebula cloud using layered noise per channel
    r_noise = layered_noise(width, height, 6, rng)
    g_noise = layered_noise(width, height, 6, rng)
    b_noise = layered_noise(width, height, 6, rng)

    # Deep space color palette: darks with purple/blue/magenta accents
    r_channel = (r_noise * 120 + rng.uniform(20, 60)).clip(0, 255)
    g_channel = (g_noise * 60  + rng.uniform(0,  30)).clip(0, 255)
    b_channel = (b_noise * 180 + rng.uniform(40, 100)).clip(0, 255)

    img_array = np.stack([r_channel, g_channel, b_channel], axis=2).astype(np.uint8)
    img = Image.fromarray(img_array)

    # Add star points: small bright dots scattered across the image
    draw = ImageDraw.Draw(img)
    n_stars = rng.integers(200, 600)
    for _ in range(n_stars):
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        brightness = rng.integers(180, 255)
        size = rng.choice([1, 1, 1, 2, 2, 3])  # mostly small
        color = (brightness, brightness, int(brightness * rng.uniform(0.8, 1.0)))
        draw.ellipse([x-size, y-size, x+size, y+size], fill=color)

    # Bright nebula core
    core_x = rng.integers(width // 4, 3 * width // 4)
    core_y = rng.integers(height // 4, 3 * height // 4)
    core_color = tuple(rng.integers(100, 200, size=3).tolist())
    draw.ellipse([core_x-50, core_y-50, core_x+50, core_y+50],
                 fill=core_color)
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return img


def generate_forest(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """
    Forest canopy looking up: fine organic texture, strong green channel,
    light filtering through leaves producing bright spots on dark background.
    Characteristic: high spatial frequency texture, green/yellow/dark palette.
    """
    # Base: dark green background with noise texture
    base_noise = layered_noise(width, height, 8, rng)
    detail_noise = layered_noise(width, height, 12, rng)

    # Forest color: dark greens with occasional yellow-green light patches
    r_channel = (base_noise * 60  + detail_noise * 30 + 10).clip(0, 255)
    g_channel = (base_noise * 120 + detail_noise * 60 + 30).clip(0, 255)
    b_channel = (base_noise * 30  + detail_noise * 20 + 5).clip(0, 255)

    img_array = np.stack([r_channel, g_channel, b_channel], axis=2).astype(np.uint8)
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Light patches filtering through canopy
    n_patches = rng.integers(10, 30)
    for _ in range(n_patches):
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        r = rng.integers(5, 40)
        brightness = rng.integers(180, 255)
        draw.ellipse([x-r, y-r, x+r, y+r],
                     fill=(brightness, brightness, int(brightness * 0.7)))

    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def generate_coral(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """
    Coral reef / underwater: high color complexity, irregular organic edges,
    saturated warm and cool colors coexisting.
    Characteristic: complex color interactions, medium-high spatial frequency.
    """
    base = layered_noise(width, height, 7, rng)
    mask = layered_noise(width, height, 4, rng)

    # Coral palette: warm oranges/pinks/reds against cool blues/teals
    r_channel = np.where(mask > 0.5,
                         base * 200 + 55,
                         base * 50 + 10).clip(0, 255)
    g_channel = np.where(mask > 0.5,
                         base * 80 + 20,
                         base * 140 + 40).clip(0, 255)
    b_channel = np.where(mask > 0.5,
                         base * 40 + 10,
                         base * 180 + 50).clip(0, 255)

    img_array = np.stack([r_channel, g_channel, b_channel], axis=2).astype(np.uint8)
    img = Image.fromarray(img_array)

    # Coral branch structures
    draw = ImageDraw.Draw(img)
    n_branches = rng.integers(5, 15)
    for _ in range(n_branches):
        x0 = rng.integers(0, width)
        y0 = rng.integers(height // 2, height)
        color = tuple(rng.integers(150, 255, size=3).tolist())
        for _ in range(rng.integers(3, 8)):
            x1 = x0 + rng.integers(-80, 80)
            y1 = y0 - rng.integers(20, 100)
            draw.line([x0, y0, x1, y1], fill=color,
                      width=rng.integers(2, 8))
            x0, y0 = x1, y1

    return img


def generate_architecture(width: int, height: int,
                           rng: np.random.Generator) -> Image.Image:
    """
    Architecture / cityscape: hard geometric lines, high contrast edges,
    regular grid-like structures.
    Characteristic: low spatial frequency base with high-contrast hard edges,
    neutral color palette with accent colors.
    """
    img = Image.new("RGB", (width, height),
                    tuple(rng.integers(30, 80, size=3).tolist()))
    draw = ImageDraw.Draw(img)

    # Building grid: vertical rectangles of varying heights
    n_buildings = rng.integers(8, 20)
    building_width = width // n_buildings

    for i in range(n_buildings):
        x0 = i * building_width
        x1 = x0 + building_width - rng.integers(2, 8)
        building_height = rng.integers(height // 4, height)
        y0 = height - building_height
        # Building face
        facade_color = tuple(rng.integers(60, 180, size=3).tolist())
        draw.rectangle([x0, y0, x1, height], fill=facade_color)

        # Windows: regular grid of small rectangles
        window_color = tuple(rng.integers(200, 255, size=3).tolist())
        for wy in range(y0 + 10, height - 10, rng.integers(15, 25)):
            for wx in range(x0 + 5, x1 - 5, rng.integers(10, 18)):
                if rng.random() > 0.3:  # some windows dark
                    draw.rectangle([wx, wy, wx+6, wy+8], fill=window_color)

    # Sky gradient at top
    sky_color = tuple(rng.integers(20, 100, size=3).tolist())
    for y in range(height // 3):
        alpha = y / (height // 3)
        r = int(sky_color[0] * (1 - alpha) + 20 * alpha)
        g = int(sky_color[1] * (1 - alpha) + 20 * alpha)
        b = int(sky_color[2] * (1 - alpha) + 60 * alpha)
        draw.line([0, y, width, y], fill=(r, g, b))

    return img


def generate_mountain(width: int, height: int,
                       rng: np.random.Generator) -> Image.Image:
    """
    Mountain landscape: strong horizontal band structure (sky/mountain/ground),
    smooth gradients with hard silhouette edges.
    Characteristic: horizontal dominance, wide dynamic range, cool palette.
    """
    img_array = np.zeros((height, width, 3), dtype=np.float32)

    # Sky: blue gradient top to horizon
    sky_height = int(height * rng.uniform(0.35, 0.55))
    for y in range(sky_height):
        t = y / sky_height
        r = int(rng.uniform(100, 150) * (1 - t) + rng.uniform(180, 220) * t)
        g = int(rng.uniform(150, 200) * (1 - t) + rng.uniform(200, 230) * t)
        b = int(rng.uniform(200, 255) * (1 - t) + rng.uniform(220, 255) * t)
        img_array[y, :] = [r, g, b]

    # Mountain silhouette using noise
    mountain_noise = layered_noise(width, 1, 5, rng).flatten()
    mountain_profile = (mountain_noise * height * 0.4 + sky_height).astype(int)
    mountain_profile = np.clip(mountain_profile, sky_height,
                                int(height * 0.85))

    mountain_color = tuple(rng.integers(60, 140, size=3).tolist())
    snow_line = int(sky_height + (mountain_profile.min() - sky_height) * 0.3)

    for x in range(width):
        peak = mountain_profile[x]
        for y in range(peak, height):
            if y < snow_line + rng.integers(-10, 10):
                img_array[y, x] = [240, 240, 245]  # snow
            else:
                img_array[y, x] = mountain_color

    # Foreground: darker ground
    ground_start = int(height * 0.75)
    ground_color = tuple(rng.integers(30, 80, size=3).tolist())
    img_array[ground_start:, :] = ground_color

    # Add some noise texture to ground
    ground_noise = layered_noise(width, height - ground_start, 8, rng)
    img_array[ground_start:, :, 1] += (ground_noise * 20).astype(np.float32)

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def generate_macro_biology(width: int, height: int,
                            rng: np.random.Generator) -> Image.Image:
    """
    Biological macro: extreme fine detail, unusual saturated colors,
    radial or cellular structures (like pollen, cell walls, insect eyes).
    Characteristic: very high spatial frequency, unusual color combinations.
    """
    img_array = np.zeros((height, width, 3), dtype=np.float32)

    # Choose between cellular and radial structure
    structure = rng.choice(["cellular", "radial"])

    if structure == "cellular":
        # Voronoi-like cellular pattern
        n_cells = rng.integers(20, 60)
        centers = rng.integers(0, [width, height], size=(n_cells, 2))
        cell_colors = rng.integers(80, 255, size=(n_cells, 3))

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        coords = np.stack([xx, yy], axis=2)

        # Assign each pixel to nearest center
        min_dist = np.full((height, width), np.inf)
        cell_assignment = np.zeros((height, width), dtype=int)

        for i, center in enumerate(centers):
            dist = np.sqrt(((coords - center) ** 2).sum(axis=2))
            mask = dist < min_dist
            min_dist[mask] = dist[mask]
            cell_assignment[mask] = i

        for i in range(n_cells):
            mask = cell_assignment == i
            img_array[mask] = cell_colors[i]

        # Cell walls: darken edges between cells
        from PIL import ImageFilter
        cell_img = Image.fromarray(img_array.astype(np.uint8))
        edges = cell_img.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges, dtype=np.float32)
        img_array = np.clip(img_array - edge_array * 0.5, 0, 255)

    else:
        # Radial pattern: concentric rings with noise
        cx, cy = width // 2, height // 2
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)

        # Normalize distance
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist_from_center / max_dist

        # Angular position for spiral effect
        angle = np.arctan2(yy - cy, xx - cx)

        # Ring pattern with angular modulation
        freq = rng.uniform(8, 20)
        n_spokes = rng.integers(6, 16)
        pattern = np.sin(dist_norm * freq * np.pi +
                         angle * n_spokes)
        pattern = (pattern + 1) / 2  # normalize to [0,1]

        # Color mapping
        base_hue = rng.integers(0, 3)
        if base_hue == 0:  # green/yellow (pollen-like)
            img_array[:,:,0] = pattern * 180 + 40
            img_array[:,:,1] = pattern * 220 + 20
            img_array[:,:,2] = pattern * 40
        elif base_hue == 1:  # blue/purple (cell-like)
            img_array[:,:,0] = pattern * 100 + 20
            img_array[:,:,1] = pattern * 60 + 10
            img_array[:,:,2] = pattern * 220 + 30
        else:  # orange/red (insect-like)
            img_array[:,:,0] = pattern * 220 + 30
            img_array[:,:,1] = pattern * 100 + 20
            img_array[:,:,2] = pattern * 40 + 10

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def generate_abstract_texture(width: int, height: int,
                               rng: np.random.Generator) -> Image.Image:
    """
    Abstract texture: pure texture with no semantic content.
    Characteristic: high spatial frequency, no recognizable structure,
    tests metric behavior on pure noise vs pure structure.
    """
    texture_type = rng.choice(["marble", "fabric", "sand"])

    if texture_type == "marble":
        # Marble: layered noise warped by a sine function
        base = layered_noise(width, height, 6, rng)
        warp = layered_noise(width, height, 4, rng)
        marble = np.sin((base + warp * 2) * np.pi * 4)
        marble = (marble + 1) / 2

        # Marble color: white/grey with colored veins
        vein_color = rng.integers(0, 200, size=3)
        r = (marble * (255 - vein_color[0]) + vein_color[0]).clip(0, 255)
        g = (marble * (255 - vein_color[1]) + vein_color[1]).clip(0, 255)
        b = (marble * (255 - vein_color[2]) + vein_color[2]).clip(0, 255)

    elif texture_type == "fabric":
        # Fabric: perpendicular sine waves at different frequencies
        x = np.linspace(0, rng.uniform(10, 30) * np.pi, width)
        y = np.linspace(0, rng.uniform(10, 30) * np.pi, height)
        xx, yy = np.meshgrid(x, y)
        warp = layered_noise(width, height, 3, rng)
        weave_h = np.sin(xx + warp * 2)
        weave_v = np.sin(yy + warp * 2)
        fabric = (weave_h * weave_v + 1) / 2

        base_color = rng.integers(50, 200, size=3)
        r = (fabric * base_color[0] + (1-fabric) * rng.integers(200, 255)).clip(0, 255)
        g = (fabric * base_color[1] + (1-fabric) * rng.integers(200, 255)).clip(0, 255)
        b = (fabric * base_color[2] + (1-fabric) * rng.integers(200, 255)).clip(0, 255)

    else:
        # Sand: fine granular noise
        fine = layered_noise(width, height, 10, rng)
        coarse = layered_noise(width, height, 3, rng)
        sand = fine * 0.6 + coarse * 0.4
        sand_color = rng.integers(150, 220, size=3)
        r = (sand * sand_color[0] + rng.uniform(10, 30)).clip(0, 255)
        g = (sand * sand_color[1] * 0.9 + rng.uniform(10, 30)).clip(0, 255)
        b = (sand * sand_color[2] * 0.6 + rng.uniform(5, 20)).clip(0, 255)

    img_array = np.stack([r, g, b], axis=2).astype(np.uint8)
    return Image.fromarray(img_array)


def generate_gradient(width: int, height: int,
                       rng: np.random.Generator) -> Image.Image:
    """
    Smooth gradient: minimal detail, tests metric behavior on smooth tonal
    transitions. Important baseline — if metrics disagree here, it reveals
    sensitivity to global tonal shifts rather than structural changes.
    Characteristic: very low spatial frequency, smooth color transitions.
    """
    gradient_type = rng.choice(["linear", "radial", "diagonal"])

    xx, yy = np.meshgrid(np.linspace(0, 1, width),
                          np.linspace(0, 1, height))

    if gradient_type == "linear":
        t = yy  # top to bottom
    elif gradient_type == "radial":
        cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        t = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        t = (t - t.min()) / (t.max() - t.min())
    else:
        angle = rng.uniform(0, np.pi)
        t = xx * np.cos(angle) + yy * np.sin(angle)
        t = (t - t.min()) / (t.max() - t.min())

    # Two endpoint colors
    color_a = rng.integers(0, 255, size=3).astype(float)
    color_b = rng.integers(0, 255, size=3).astype(float)

    r = (t * color_b[0] + (1 - t) * color_a[0]).clip(0, 255)
    g = (t * color_b[1] + (1 - t) * color_a[1]).clip(0, 255)
    b = (t * color_b[2] + (1 - t) * color_a[2]).clip(0, 255)

    img_array = np.stack([r, g, b], axis=2).astype(np.uint8)
    return Image.fromarray(img_array)


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

CATEGORIES = {
    "nebula":           generate_nebula,
    "forest":           generate_forest,
    "coral":            generate_coral,
    "architecture":     generate_architecture,
    "mountain":         generate_mountain,
    "macro_biology":    generate_macro_biology,
    "abstract_texture": generate_abstract_texture,
    "gradient":         generate_gradient,
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_all(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    total = len(CATEGORIES) * len(ASPECT_RATIOS) * REPLICATES
    generated = 0
    failed = []

    print(f"\n{'='*60}")
    print(f"Generating synthetic_v2 test set")
    print(f"Categories:    {len(CATEGORIES)}")
    print(f"Aspect ratios: {len(ASPECT_RATIOS)}")
    print(f"Replicates:    {REPLICATES}")
    print(f"Total images:  {total}")
    print(f"Output dir:    {output_dir}")
    print(f"{'='*60}\n")

    for category_name, generator_fn in CATEGORIES.items():
        cat_dir = os.path.join(output_dir, category_name)
        os.makedirs(cat_dir, exist_ok=True)

        for aspect_name, width, height in ASPECT_RATIOS:
            for replicate in range(1, REPLICATES + 1):

                # Unique seed per cell so replicates differ but are reproducible
                seed = BASE_SEED + hash(f"{category_name}_{aspect_name}_{replicate}") % 100000
                rng = np.random.default_rng(seed)

                filename = f"{category_name}_{aspect_name}_r{replicate}.png"
                filepath = os.path.join(cat_dir, filename)

                try:
                    img = generator_fn(width, height, rng)
                    img.save(filepath)
                    generated += 1
                    print(f"  [{generated:>3}/{total}] {filename} "
                          f"({width}×{height})")
                except Exception as e:
                    print(f"  ERROR generating {filename}: {e}")
                    failed.append(filename)

    print(f"\n{'='*60}")
    print(f"Generated: {generated}/{total}")
    if failed:
        print(f"Failed:    {len(failed)} — {failed}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic_v2 test image set"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()
    generate_all(args.output_dir)