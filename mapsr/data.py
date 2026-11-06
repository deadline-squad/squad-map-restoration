"""Aligned training pairs: the whole-map texture, the geometry render and a layer's own finer texture on one grid.

A Seed or Skirmish layer of the game ships a minimap texture of its own over a part of the map, 2 to 5 times finer than the
whole-map texture. Each pair is written to `<PAIRS>/<tag>.npz` on a grid whose step is half the whole-map texture's metres per
pixel:

    base   the whole-map texture, bicubic           (what a map viewer shows today)
    guide  the Detailed tiles                       (geometry from the game files: buildings, roads, trees, water)
    truth  the layer's own texture                  (the target)
    mask   inside the layer's play-area border, eroded (outside it the game dims the texture)

    python -m mapsr.data pairs.json
"""
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import config
from .bc1 import decode_bc1

ERODE_M = 30  # the play-area border is shrunk by this much: the texture is dimmed with a soft edge around it


def load_texture(name):
    """RGB uint8 array of an extracted minimap texture, `<name>.PF_DXT1.<w>x<h>.bin` or `<name>.PF_B8G8R8A8.<w>x<h>.bin`."""
    path = next(f for f in os.listdir(config.TEXTURES) if f.startswith(name + ".PF_"))
    _, fmt, size, _ = path.split(".")
    width, height = (int(v) for v in size.split("x"))
    with open(os.path.join(config.TEXTURES, path), "rb") as handle:
        raw = handle.read()
    if fmt == "PF_DXT1":
        return decode_bc1(raw, width, height)
    if fmt == "PF_B8G8R8A8":
        return np.frombuffer(raw, np.uint8).reshape(height, width, 4)[..., [2, 1, 0]].copy()
    raise ValueError(f"unsupported pixel format {fmt}")


def tile_mosaic(slug, zoom, x0, y0, x1, y1):
    """Detailed tiles `<TILES>/<slug>/<zoom>/<x>_<y>.webp` over the world rectangle (m).

    Returns the mosaic, the world coordinates (m) of its top-left corner and its step (m per px) along x and y.
    """
    frame = config.load_maps()[slug]
    size_x, size_y = frame["size"]
    pixels = 256 * 2 ** zoom
    px = lambda x: (x - frame["min"][0]) / size_x * pixels
    py = lambda y: (y - frame["min"][1]) / size_y * pixels
    tx0, tx1 = int(px(x0) // 256), int(px(x1) // 256)
    ty0, ty1 = int(py(y0) // 256), int(py(y1) // 256)
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            path = os.path.join(config.TILES, slug, str(zoom), f"{tx}_{ty}.webp")
            if os.path.exists(path):
                mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = np.asarray(Image.open(path).convert("RGB"))
    origin = (frame["min"][0] + tx0 * 256 * size_x / pixels, frame["min"][1] + ty0 * 256 * size_y / pixels)
    return mosaic, origin, (size_x / pixels, size_y / pixels)


def resample(image, src_origin, src_step, grid_origin, grid_step, grid_size, resample_filter):
    """`image` (origin and step in m) on a grid, with Pillow's exact box resize; the box is clamped to the image."""
    box = (
        (grid_origin[0] - src_origin[0]) / src_step[0],
        (grid_origin[1] - src_origin[1]) / src_step[1],
        (grid_origin[0] - src_origin[0] + grid_size[0] * grid_step[0]) / src_step[0],
        (grid_origin[1] - src_origin[1] + grid_size[1] * grid_step[1]) / src_step[1],
    )
    height, width = image.shape[:2]
    box = (max(box[0], 0.0), max(box[1], 0.0), min(box[2], float(width)), min(box[3], float(height)))
    return np.asarray(Image.fromarray(image).resize(grid_size, resample_filter, box=box))


def edges(image):
    """Gradient magnitude of the smoothed luminance."""
    smooth = ndimage.gaussian_filter(image.astype(np.float32).mean(-1), 1.0)
    return np.hypot(ndimage.sobel(smooth, 1), ndimage.sobel(smooth, 0))


def ncc(a, b):
    """Normalised cross-correlation of two equally shaped arrays."""
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def fetch_border(layer):
    """Play-area polygon of a layer in world metres, from the layer data of the SquadCalc API."""
    request = urllib.request.Request(f"{config.LAYER_API}?name={layer}", headers={"X-App-Version": "45.34.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    return [(p["location_x"] / 100, p["location_y"] / 100) for p in data["border"]]


def build_pair(slug, base_name, layer_name, corners_cm, layer, tag):
    """Write `<PAIRS>/<tag>.npz` and return a summary.

    corners_cm: the layer's MapTextureCorner0/1 as (x0, y0, x1, y1) in centimetres; layer: its name in the API.
    The layer's texture is registered on the whole-map texture by edge correlation (shift reported in whole-map pixels).
    """
    frame = config.load_maps()[slug]
    size_x, size_y = frame["size"]
    base = load_texture(base_name)
    truth = load_texture(layer_name)
    lx0, lx1 = sorted((corners_cm[0] / 100, corners_cm[2] / 100))
    ly0, ly1 = sorted((corners_cm[1] / 100, corners_cm[3] / 100))
    base_step = (size_x / base.shape[1], size_y / base.shape[0])
    step = (base_step[0] / 2, base_step[1] / 2)
    size = (int(round((lx1 - lx0) / step[0])), int(round((ly1 - ly0) / step[1])))
    origin = (lx0, ly0)
    truth_step = ((lx1 - lx0) / truth.shape[1], (ly1 - ly0) / truth.shape[0])
    truth_grid = resample(truth, origin, truth_step, origin, step, size, Image.LANCZOS)

    def base_shifted(dx, dy, divisor, resample_filter):
        """The whole-map texture on the layer's grid (`divisor` times coarser), shifted by (dx, dy) whole-map pixels."""
        grid_step = (step[0] * divisor, step[1] * divisor)
        shifted = (origin[0] + dx * base_step[0], origin[1] + dy * base_step[1])
        return resample(base, frame["min"], base_step, shifted, grid_step, (size[0] // divisor, size[1] // divisor), resample_filter)

    def correlation(dx, dy, divisor):
        grid_step = (step[0] * divisor, step[1] * divisor)
        coarse_truth = resample(truth, origin, truth_step, origin, grid_step, (size[0] // divisor, size[1] // divisor), Image.LANCZOS)
        return ncc(edges(base_shifted(dx, dy, divisor, Image.BILINEAR)), edges(coarse_truth))

    # coarse: +-8 whole-map px in 1 px steps on a quarter-size grid; fine: +-1 px around it in 0.25 px steps on a half-size grid
    best = (-1.0, 0.0, 0.0)
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            score = correlation(float(dx), float(dy), 4)
            if score > best[0]:
                best = (score, float(dx), float(dy))
    centre_x, centre_y = best[1], best[2]
    for dy in np.arange(-1.0, 1.01, 0.25):
        for dx in np.arange(-1.0, 1.01, 0.25):
            score = correlation(centre_x + dx, centre_y + dy, 2)
            if score > best[0]:
                best = (score, centre_x + dx, centre_y + dy)
    score, shift_x, shift_y = best
    base_grid = base_shifted(shift_x, shift_y, 1, Image.BICUBIC)

    zoom = next(z for z in range(9) if size_x / (256 * 2 ** z) <= step[0])
    mosaic, mosaic_origin, mosaic_step = tile_mosaic(slug, zoom, lx0, ly0, lx1, ly1)
    guide_grid = resample(mosaic, mosaic_origin, mosaic_step, origin, step, size, Image.BILINEAR)

    polygon = [((x - lx0) / step[0], (y - ly0) / step[1]) for x, y in fetch_border(layer)]
    border = Image.new("L", size, 0)
    ImageDraw.Draw(border).polygon(polygon, fill=255)
    mask = ndimage.distance_transform_edt(np.asarray(border) > 0) > (ERODE_M / step[0])

    os.makedirs(config.PAIRS, exist_ok=True)
    np.savez(os.path.join(config.PAIRS, f"{tag}.npz"), base=base_grid, guide=guide_grid, truth=truth_grid, mask=mask)
    return {"tag": tag, "grid": size, "step_m": round(step[0], 4), "registration_corr": round(score, 3),
            "shift_base_px": (shift_x, shift_y), "tile_zoom": zoom, "mask_px": int(mask.sum())}


if __name__ == "__main__":
    with open(sys.argv[1]) as handle:
        specs = json.load(handle)
    for spec in specs:
        print(json.dumps(build_pair(**spec)), flush=True)
