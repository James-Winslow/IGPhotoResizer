# Field Photography Protocol: Instagram Empirical Verification

This document is the complete guide for collecting real-world photographs
for the IGPhotoResizer experiment. It covers what to shoot, how to shoot it,
how to upload and download, and how to document each session.

The goal is a dataset of real photographs where we have the uncompressed
original directly from the camera, so we can measure exactly what Instagram
does to each image — not just what our simulation predicts.

---

## Why Real Photos Matter

Our v2 experiment used synthetic images. The results are rigorous but
reviewers will always ask: does this hold on real photographs?

The key advantage of field photography over downloaded stock photos:
with your own photos, the "original" is the uncompressed camera file (RAW
or highest-quality JPEG from the camera). With stock photos, the original
is already a previously-compressed JPEG — introducing one extra round of
quality loss we cannot measure or control.

---

## Instagram Account Setup

Use a dedicated test account, not your personal account.

**Account settings to configure before any uploads:**
1. Settings > Account > Cellular Data Use > Use Less Data: OFF
2. Settings > Account > High Quality Uploads: ON
3. Use the same device for all uploads in a session
4. Upload over WiFi, not cellular, for consistent compression behavior
5. Document the device model and iOS/Android version in each session log

**Account name suggestion:** something neutral that does not attract
followers or algorithm attention. A private account is fine.

---

## What to Shoot: The Eight Categories

Each category maps to one of our eight synthetic image categories.
The goal is to capture images that are representative of the category's
defining visual properties — the properties our experiment showed matter
most for compression sensitivity.

### Category 1: Architecture
**Defining property:** Hard geometric edges, repeating grid patterns,
high spatial frequency in the horizontal and vertical directions.
**What to shoot:** Building facades, window grids, brick walls,
staircases, structural steel, bridges. The closer the better — fill
the frame with the geometric structure.
**Why it matters:** Architecture showed the largest effect in our
experiment (Cohen's d = 1.90). This category is the most important to
validate on real images.
**Examples:** Downtown building facades shot straight on, not at an angle.
Office tower windows. Repetitive architectural details.

### Category 2: Mountain / Landscape
**Defining property:** Strong horizontal band structure (sky, mountain,
ground), large smooth regions, some texture in vegetation or rock.
**What to shoot:** Any landscape with a clear horizon and distinct
horizontal zones. Mountain silhouettes against sky. Open fields.
**Why it matters:** Second largest effect (d = 1.61).

### Category 3: Macro Biology
**Defining property:** Organic cellular or radial structures, fine detail,
complex textures at small scale.
**What to shoot:** Close-up of leaves (veins), bark texture, flower petals,
fabric weave, rough stone. Any highly textured organic surface shot close enough
to fill the frame.
**Why it matters:** Strong effect (d = 1.01). Fine organic detail is
exactly what double-resize destroys.

### Category 4: Abstract Texture
**Defining property:** Repetitive fine texture without strong directional
structure. No dominant edges or shapes.
**What to shoot:** Concrete walls, sand, gravel, carpet, woven fabric
from a distance. Any surface where the texture is the subject.

### Category 5: Coral / Color Complexity
**Defining property:** High color diversity, irregular organic edges,
moderate spatial frequency.
**What to shoot:** Colorful flowers, markets, fruit stands, fabric
displays. Dense scenes with many colors.

### Category 6: Forest / Fine Organic Texture
**Defining property:** Fine repeated organic texture — leaves, grass,
pine needles. Moderate spatial frequency with organic irregularity.
**What to shoot:** Dense foliage, grass fields, leaf canopy from below.

### Category 7: Nebula / Soft Color Fields
**Defining property:** Smooth color gradients, soft focus, no hard edges.
**What to shoot:** Sky at golden hour, intentionally defocused lights at
night (bokeh), fog, smoke, soft reflections in water.
**Why it matters:** This is our control category — the one where our
experiment found no significant effect. Real images should confirm this.

### Category 8: Gradient
**Defining property:** Pure smooth tonal transitions. Our synthetic gradient
was completely smooth by construction. Real-world approximations are harder
to find.
**What to shoot:** Clear blue sky (no clouds), flat painted walls,
color-corrected surfaces. The goal is minimal texture.
**Why it matters:** This is our null condition. If we find an effect here,
something is wrong with the experiment.

---

## How Many Photos to Take

**Minimum viable dataset:** 3 photos per category = 24 photos total.
This matches our synthetic design (3 replicates per cell) and gives us
enough for statistical comparison.

**Preferred dataset:** 5 photos per category per aspect ratio = 200 photos.
This gives enough power for the full factorial analysis and matches
the synthetic v2 design exactly.

**Practical recommendation:** Start with 3 per category (24 photos) and
run the analysis. If the results replicate the synthetic findings, you
have enough. If they don't, collect more.

---

## Aspect Ratio Coverage

For each category, try to capture images at multiple natural aspect ratios.
Do not crop artificially — the goal is images that naturally have that shape.

| Aspect ratio | Shape | How to get it |
|---|---|---|
| Very wide (2:1) | Panoramic | Horizontal sweep, wide landscape |
| Wide (4:3) | Standard landscape | Default camera landscape orientation |
| Square (1:1) | Square | Camera in square mode, or crop post-capture |
| Portrait (3:4) | Standard portrait | Default camera portrait orientation |
| Very tall (1:2) | Tall narrow | Camera in portrait mode with vertical subject |

If a natural very wide or very tall shot is not available for a category,
it is fine to skip that combination. Do not force an unnatural crop.

---

## Camera Settings

The goal is the highest quality original possible.

**If shooting RAW:** Shoot RAW + JPEG. Keep both. The JPEG is for quick
viewing; the RAW is the ground truth original.

**If shooting JPEG only:** Set camera to highest quality, lowest compression
setting. On most cameras this is "Fine" or "SF" (Super Fine).

**Do not:**
- Apply filters, sharpening, or processing in the camera
- Use portrait mode or computational photography modes
- Apply HDR unless it is entirely hardware-based

**Do:**
- Shoot in good light where possible (reduces noise, which is its own
  source of texture that could affect results)
- Keep ISO as low as possible for the same reason
- Use a tripod for architecture shots if available (sharper edges)

---

## Upload Protocol

Consistency matters here. Use the same procedure for every upload.

**Per upload session:**

1. Transfer originals from camera to computer first
2. Do not edit, resize, or process the images before upload
3. Note the original file size and dimensions (use `python src/measure_originals.py`)
4. Open Instagram on desktop (instagram.com in Chrome)
5. Create a new post
6. For carousel posts: upload images in a specific documented order
7. Select aspect ratio in Instagram (do not let it auto-crop — confirm your
   intended aspect ratio is selected)
8. No filters. No edits. Post.
9. Note the post URL
10. Wait 5 minutes before downloading (give Instagram time to finish processing)
11. Download using instaloader (see below)
12. Verify the download hash (see below)
13. Log everything in the session file (see template below)

**Upload from desktop, not mobile.** Community evidence suggests desktop
upload applies less aggressive compression. We want to test this separately
but use desktop as the default.

---

## Download Protocol

**Install instaloader:**
```
pip install instaloader
```

**Download your own post:**
```
instaloader --login YOUR_USERNAME -- -YOUR_POST_SHORTCODE
```

The post shortcode is the last part of the post URL:
https://www.instagram.com/p/SHORTCODE/

**Verify the download:**
Download twice and compare hashes to confirm the file is stable:
```python
import hashlib

def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

hash1 = file_hash('download_1/image.jpg')
hash2 = file_hash('download_2/image.jpg')
print('Match:', hash1 == hash2)
```

If the hashes match, the download is stable. If they differ, Instagram
is serving different versions — document this and download a third time.

**Never use screenshots.** A screenshot is a re-encoded PNG of whatever
your screen rendered, not the actual file Instagram is storing. It
introduces a second compression step and changes the resolution.

---

## Session Log Template

Create one file per upload session in `experiments/instagram_empirical/`.
Name it `session_YYYYMMDD.md`.

```markdown
# Instagram Upload Session — YYYY-MM-DD

## Environment
- Device: [e.g., MacBook Pro M2, Chrome 120]
- Connection: WiFi
- Instagram account: [account name]
- High Quality Uploads: ON/OFF
- Time: HH:MM local

## Images Uploaded

### Post 1
- Filename: [original filename]
- Original dimensions: W x H px
- Original file size: X KB
- Category: [architecture / mountain / etc.]
- Aspect ratio: [very_wide / wide / square / portrait / very_tall]
- Post type: single / carousel position N of M
- Post URL: https://www.instagram.com/p/SHORTCODE/
- Upload time: HH:MM

### Downloaded
- Wait time before download: X minutes
- Download method: instaloader
- Downloaded dimensions: W x H px
- Downloaded file size: X KB
- Hash match (two downloads): YES / NO
- SSIM vs original: [run after download]
- PSNR vs original: [run after download]

## Notes
[Anything unusual — server errors, unexpected crops, quality settings etc.]
```

---

## Analysis After Each Session

After downloading, run the session analysis script to measure what
Instagram actually did:

```powershell
python src/measure_instagram_session.py --session experiments/instagram_empirical/session_YYYYMMDD.md
```

This script (to be written in v3) will:
1. Load each original and downloaded pair
2. Compute SSIM, PSNR, MSE
3. Estimate the JPEG quality factor of the downloaded file
4. Compare against our Q75 simulation
5. Write results to the session log

---

## What We Are Trying to Learn

After collecting and analyzing the sessions, we want answers to:

**Primary questions:**
1. What JPEG quality factor does Instagram actually use? (Compare to our Q75 assumption)
2. Does quality differ between single posts and carousel posts?
3. Does quality differ between carousel position 1 and position 2+?
4. Does desktop upload apply different compression than mobile?

**Secondary questions:**
5. Is compression deterministic? (Upload same image twice, compare outputs)
6. Does High Quality Uploads actually change the output?
7. Is download via instaloader lossless? (Compare hash of two downloads)
8. Does the category interaction pattern from synthetic images replicate
   on real photographs?

---

## The Lab Notebook

Document the data collection as a lab notebook — one markdown file per
session, stored in `experiments/instagram_empirical/`. Include:

- The original photo (resized for display)
- The downloaded Instagram version (side by side)
- The difference map (absolute pixel difference, contrast-enhanced)
- The SSIM heatmap
- Notes on what you observed in the field when shooting

This documentation serves three purposes:
1. Scientific record (reproducibility)
2. Portfolio content (shows systematic thinking)
3. Future training data metadata (when we build the ML model)

The progression from session 1 to session N is a story worth telling in
the eventual LinkedIn article and paper.

---

*Last updated: March 2026*
*Status: Protocol finalized. Ready to begin field collection.*
