# squad-map-restoration

Guided x2 restoration of the whole-map minimap textures of Squad, trained on the finer textures the game ships for some of its
layers.

## Why

The in-game map of a regular layer is one 4096 x 4096 texture over the whole map (0.67 m per pixel on Mutaha), and its roofs are
clipped to white. A Seed or Skirmish layer ships a texture of its own over a part of the map (Mutaha Seed v1: 4096 px over 1300 m,
0.32 m per pixel) with the real roof colours. The game has no finer texture for a regular layer, so a map viewer that zooms in
shows blocks. This repository trains a small net that restores the whole-map texture towards what the finer capture of the same
place looks like, using the geometry of the game files (buildings, roads, trees, water) as a guide.

## Method

Per pair, on one grid whose step is half the whole-map texture's metres per pixel (Mutaha: 0.3363 m):

| | |
|---|---|
| `base` | the whole-map texture, bicubic x2 |
| `guide` | the Detailed tiles of zoom 5 (a render of the game's geometry: building footprints and heights, roads, tree crowns, water; the labels and contours of that render are in it) |
| `truth` | the layer's own texture, Lanczos to the grid, registered on `base` by edge correlation (coarse +-8 px, fine 0.25 px) |
| `mask` | inside the layer's play-area border, eroded by 30 m (outside it the game dims the texture) |

The net (`mapsr/model.py`) is an EDSR-style residual CNN without batch norm: 3x3 conv 6 -> 48, 8 residual blocks (conv, ReLU,
conv, scaled by 0.5), 3x3 conv 48 -> 3, 336 483 parameters. It predicts a correction to the bicubic texture:
`output = base + tail(features)`. Loss: L1 on colour plus 0.5 x L1 on the horizontal and vertical gradients. Adam, learning rate
2e-4 with a cosine decay to 0.1x, batches of 16 patches of 96 px, rotation by 90 degrees and flips, fp32 (the GTX 1650 has no
tensor cores and fp16 is 4x slower there). The southern quarter of the play area, with a 128 px gap, is held out for validation.

## Results

One run, trained on Mutaha Seed v1 only (3690 steps, 25 min on a GTX 1650), `python -m mapsr.evaluate model.pt mutaha_seed tallil_seed`.
The first set is the held-out southern quarter of the training map, the second a whole map the net never saw. Detail/truth is
the mean |Laplacian| relative to the truth (1.0 = the same amount of fine detail).

| set | image | PSNR | SSIM | edge F1 | detail/truth |
|---|---|---|---|---|---|
| Mutaha Seed, held-out south | bicubic | 17.32 | 0.633 | 0.773 | 0.56 |
| | net | 23.42 | 0.685 | 0.859 | 0.25 |
| Tallil Seed, never seen | bicubic | 16.20 | 0.426 | 0.634 | 0.99 |
| | net | 16.23 | 0.482 | 0.546 | 0.45 |

What the numbers and the before/after crops show:

- The net removes the block artefacts of the whole-map texture and corrects the colour of roofs and ground. That is where
  the PSNR gain on Mutaha comes from.
- It does not recover fine detail: the result is smoother than the truth (detail 0.25 of the truth's), and trees, roof
  structure and road markings of the Seed texture do not come back.
- It does not generalise yet: on Tallil the PSNR is unchanged, the edge F1 drops from 0.634 to 0.546 and the output has less
  than half of the detail of the truth.

This model is not good enough to replace the whole-map texture in a viewer. See the open issues for the next steps.

## Inputs (not in this repository)

All of them derive from game assets and are not published here.

- `textures/`: the minimap textures of the game client as raw mip 0, `<name>.PF_DXT1.4096x4096.bin` (BC1) or
  `<name>.PF_B8G8R8A8.<w>x<h>.bin`, extracted with CUE4Parse (`MapTexture` of each layer's world settings).
- `tiles/<slug>/<zoom>/<x>_<y>.webp`: the Detailed tiles (the deployment's `tools/maptiles`), the guide.
- `maps.json`: the map frames, from `node scripts/export_maps_json.mjs <path to SquadCalc>/src/data/maps.js > maps.json`.
- The play-area border of a layer comes from the layer data of the SquadCalc API (`MAPSR_LAYER_API`).

Paths are set with `MAPSR_TEXTURES`, `MAPSR_TILES`, `MAPSR_MAPS`, `MAPSR_PAIRS` and `MAPSR_LAYER_API`
(`mapsr/config.py`).

## Use

```sh
pip install -r requirements.txt
python -m unittest discover -s tests

python -m mapsr.data pairs.json                                  # aligned pairs into pairs/<tag>.npz
python -m mapsr.train --train mutaha_seed --hold mutaha_seed --val tallil_seed --minutes 25 --out model.pt --shots shots
python -m mapsr.evaluate model.pt mutaha_seed tallil_seed        # PSNR, SSIM, edge F1, detail ratio
python -m mapsr.infer mutaha Mutaha_Minimap model.pt out/mutaha-hd --png preview.png   # 256 px WebP tiles z0..5
```

`pairs.json` is a list of `{"slug", "base_name", "layer_name", "corners_cm": [x0, y0, x1, y1], "layer", "tag"}`;
`corners_cm` are the layer's `MapTextureCorner0/1` in centimetres.

## Limits

- Only Mutaha Seed v1 is a clean pair (same art style and content as the whole-map texture). Skirmish textures are drawn in
  another style; the Skorpo, Tallil, Goose Bay and Manicouagan pairs differ in tone or content (water, grading), and the
  registration of Black Coast failed. One map, about 2.2 million masked pixels, so the net may not generalise to snow or
  forest maps, and overfitting is checked only on the held-out southern quarter.
- The guide is a render, not ground truth: on Manicouagan it marks a field as water.
- PSNR gains come mostly from correcting the roof tone; structure is judged with SSIM, edge F1 and the detail ratio
  (`mapsr.evaluate`), not by PSNR alone. An L1 loss regresses to a smooth mean, which is why the detail ratio is low.
- The output is a restoration by a trained model, not a capture: fine detail that no input contains is not recovered.

## Data and weights

The textures, the tiles, the pairs and the trained weights are derived from Squad game assets. They are kept out of version
control (`.gitignore`) and should not be published.
