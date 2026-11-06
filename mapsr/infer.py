"""Restore a whole map and write 256 px WebP tiles z0..5 (z5 = 8192 px over the map frame, the grid of the Detailed tiles).

    python -m mapsr.infer <slug> <whole-map texture name> <model.pt> <out dir> [--png preview.png]

Per pixel of the z5 grid the net gets the whole-map texture (bicubic x2 to 8192 px) and the Detailed tile of zoom 5.
`<out dir>/meta.json` follows the tile folders of the Detailed layer. The folder is written next to its final name and
swapped in when complete.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from . import config
from .data import load_texture, tile_mosaic
from .model import Net, device, restore

GRID = 8192  # zoom 5: 256 * 2 ** 5 px over the map frame
SOURCES = "the game's whole-map minimap texture restored with the geometry of the Detailed tiles by a net trained on Seed layer textures of the game"


def zoom5_guide(slug):
    """The Detailed tiles of zoom 5 over the whole map frame: (8192, 8192, 3)."""
    frame = config.load_maps()[slug]
    mosaic, _, _ = tile_mosaic(slug, 5, frame["min"][0], frame["min"][1], frame["min"][0] + frame["size"][0] - 1e-6, frame["min"][1] + frame["size"][1] - 1e-6)
    return mosaic[:GRID, :GRID]


def write_tiles(image, folder, quality=90):
    """z5 tiles from `image` and the coarser zooms by 2 x 2 averaging, as `<folder>/<z>/<x>_<y>.webp`."""
    level = image
    for zoom in range(5, -1, -1):
        count = level.shape[0] // 256
        os.makedirs(os.path.join(folder, str(zoom)), exist_ok=True)
        for tx in range(count):
            for ty in range(count):
                tile = level[ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256]
                Image.fromarray(tile).save(os.path.join(folder, str(zoom), f"{tx}_{ty}.webp"), quality=quality, method=4)
        if zoom:
            level = level.reshape(level.shape[0] // 2, 2, level.shape[1] // 2, 2, 3).mean((1, 3)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("base")
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--png", default="", help="also write a 2048 px preview")
    args = parser.parse_args()

    started = time.time()
    frame = config.load_maps()[args.slug]
    net = Net().to(device())
    net.load_state_dict(torch.load(args.model, map_location=device()))
    base = np.asarray(Image.fromarray(load_texture(args.base)).resize((GRID, GRID), Image.BICUBIC))
    restored = (restore(net, base, zoom5_guide(args.slug), tile=512, pad=32) + 0.5).astype(np.uint8)
    print(f"{args.slug}: net done in {time.time() - started:.0f} s", flush=True)
    if args.png:
        Image.fromarray(restored).resize((2048, 2048), Image.LANCZOS).save(args.png)

    temporary = args.out + ".tmp"
    write_tiles(restored, temporary)
    meta = {
        "map": args.slug, "style": "hd", "tileSize": 256, "minNativeZoom": 0, "maxNativeZoom": 5,
        "metresPerPixel": round(frame["size"][0] / GRID, 4),
        "bounds_m": {"min_x": frame["min"][0], "max_x": frame["min"][0] + frame["size"][0], "min_y": frame["min"][1], "max_y": frame["min"][1] + frame["size"][1]},
        "url": "{z}/{x}_{y}.webp", "sources": SOURCES, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(temporary, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    if os.path.exists(args.out):
        os.rename(args.out, args.out + ".old")
    os.rename(temporary, args.out)
    print(f"{args.slug}: tiles written in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
