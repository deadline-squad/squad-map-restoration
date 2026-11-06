"""Train the net on the pairs of `mapsr.data` for a fixed time (fp32: the GTX 1650 has no tensor cores and fp16 is 4x slower).

    python -m mapsr.train --train mutaha_seed --hold mutaha_seed --val tallil_seed --minutes 25 --out model.pt

`--hold <tag>`: the southern quarter of that pair's play area (with a 128 px gap) is kept out of training and used for
validation. `--val <tag>,...`: whole pairs that are never trained on.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from . import config
from .metrics import psnr, ssim
from .model import Net, device, restore

LEARNING_RATE = 2e-4
HOLDOUT_GAP = 128


def load_pair(tag):
    with np.load(os.path.join(config.PAIRS, f"{tag}.npz")) as data:
        return {key: data[key] for key in ("base", "guide", "truth", "mask")}


def split_south(mask):
    """Boolean masks (train, validation): the southern quarter of the play area validates, a gap of HOLDOUT_GAP rows separates."""
    rows = np.where(mask.any(1))[0]
    cut = int(rows.min() + 0.75 * (rows.max() - rows.min()))
    train, validation = mask.copy(), mask.copy()
    train[cut - HOLDOUT_GAP:] = False
    validation[:cut] = False
    return train, validation


def centres(mask, half):
    """Row and column of every pixel whose square of side 2 * half lies inside the mask."""
    return np.where(ndimage.binary_erosion(mask, iterations=half))


def sample_batch(sets, size, count, rng, where):
    """`count` random patches of `size` px (random rotation by 90 degrees and flip): (input 6 channels, target 3), 0..1."""
    inputs, targets = [], []
    half = size // 2
    for _ in range(count):
        chosen = sets[rng.integers(len(sets))]
        i = rng.integers(len(chosen["rows"]))
        window = (slice(chosen["rows"][i] - half, chosen["rows"][i] + half), slice(chosen["cols"][i] - half, chosen["cols"][i] + half))
        x = np.concatenate([chosen["pair"]["base"][window], chosen["pair"]["guide"][window]], -1)
        y = chosen["pair"]["truth"][window]
        turns = rng.integers(4)
        x, y = np.rot90(x, turns), np.rot90(y, turns)
        if rng.integers(2):
            x, y = x[:, ::-1], y[:, ::-1]
        inputs.append(x.copy())
        targets.append(y.copy())
    to_tensor = lambda arrays: torch.from_numpy(np.stack(arrays)).to(where).permute(0, 3, 1, 2).float() / 255
    return to_tensor(inputs), to_tensor(targets)


def loss_of(predicted, target):
    """L1 on colour plus half the L1 on the horizontal and vertical gradients (keeps edges from washing out)."""
    dx = lambda v: v[..., :, 1:] - v[..., :, :-1]
    dy = lambda v: v[..., 1:, :] - v[..., :-1, :]
    colour = F.l1_loss(predicted, target)
    return colour + 0.5 * (F.l1_loss(dx(predicted), dx(target)) + F.l1_loss(dy(predicted), dy(target))), colour


def evaluate(net, name, pair, mask):
    """Print PSNR and SSIM of the bicubic input and of the net over the mask; returns the prediction and its top-left corner."""
    rows, cols = np.where(mask)
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    region = mask[y0:y1, x0:x1]
    truth = pair["truth"][y0:y1, x0:x1].astype(np.float32)
    base = pair["base"][y0:y1, x0:x1].astype(np.float32)
    predicted = restore(net, pair["base"], pair["guide"], (y0, y1, x0, x1))
    print(f"  {name:28s} bicubic PSNR {psnr(base, truth, region):5.2f} SSIM {ssim(base, truth, region):.4f} | "
          f"net PSNR {psnr(predicted, truth, region):5.2f} SSIM {ssim(predicted, truth, region):.4f}", flush=True)
    return predicted, (y0, x0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="comma-separated pair tags")
    parser.add_argument("--hold", default="", help="pair whose southern quarter is held out")
    parser.add_argument("--val", default="", help="comma-separated pair tags never trained on")
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="model.pt")
    parser.add_argument("--shots", default="", help="directory for before / after / truth crops of the validation sets")
    args = parser.parse_args()

    where = device()
    rng = np.random.default_rng(1)
    training, held = [], None
    for tag in args.train.split(","):
        pair = load_pair(tag)
        mask = pair["mask"]
        if tag == args.hold:
            mask, validation_mask = split_south(mask)
            held = (pair, validation_mask)
        rows, cols = centres(mask, args.patch // 2 + 2)
        print(f"{tag}: {len(rows)} training centres", flush=True)
        training.append({"pair": pair, "rows": rows, "cols": cols})
    unseen = [(tag, load_pair(tag)) for tag in args.val.split(",") if tag]

    torch.backends.cudnn.benchmark = True
    net = Net().to(where)
    optimiser = torch.optim.Adam(net.parameters(), LEARNING_RATE)
    print("parameters", sum(p.numel() for p in net.parameters()), flush=True)
    started = last_report = time.time()
    seconds, step = args.minutes * 60, 0
    while time.time() - started < seconds:
        inputs, targets = sample_batch(training, args.patch, args.batch, rng, where)
        learning_rate = LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (time.time() - started) / seconds)))
        for group in optimiser.param_groups:
            group["lr"] = learning_rate
        loss, colour = loss_of(net(inputs), targets)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        step += 1
        if time.time() - last_report > 120:
            last_report = time.time()
            print(f"step {step} {step / (time.time() - started):.1f} it/s loss {loss.item():.4f} l1 {colour.item():.4f} lr {learning_rate:.2e}", flush=True)
    torch.save(net.state_dict(), args.out)
    print(f"trained {step} steps in {(time.time() - started) / 60:.1f} min", flush=True)

    print("validation:", flush=True)
    results = []
    if held:
        results.append(("hold_" + args.hold, held[0], *evaluate(net, "held-out south of " + args.hold, held[0], held[1])))
    for tag, pair in unseen:
        results.append(("val_" + tag, pair, *evaluate(net, "unseen map " + tag, pair, pair["mask"])))
    if args.shots:
        os.makedirs(args.shots, exist_ok=True)
        for name, pair, predicted, (y0, x0) in results:
            height, width = predicted.shape[:2]
            window = (slice(max(height // 2 - 250, 0), height // 2 + 250), slice(max(width // 2 - 250, 0), width // 2 + 250))
            region = (slice(y0, y0 + height), slice(x0, x0 + width))
            row = np.concatenate([pair["base"][region][window], np.clip(predicted[window], 0, 255).astype(np.uint8), pair["truth"][region][window]], 1)
            Image.fromarray(row).save(os.path.join(args.shots, f"{name}.png"))


if __name__ == "__main__":
    main()
