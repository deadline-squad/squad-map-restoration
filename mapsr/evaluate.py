"""Structure metrics of a trained net on the held-out south of a pair and on whole unseen pairs.

    python -m mapsr.evaluate <model.pt> <hold tag> [<unseen tag> ...]

Per set, for the bicubic baseline and the net: PSNR, SSIM, edge F1 against the truth (1 px tolerance) and the ratio of the
mean |Laplacian| to the truth's (1.0 = the same amount of fine detail).
"""
import sys

import numpy as np
import torch

from .metrics import detail, edge_f1, psnr, ssim
from .model import Net, device, restore
from .train import load_pair, split_south


def report(name, base, predicted, truth, mask):
    truth_detail = detail(truth, mask)
    for label, image in (("bicubic", base), ("net", predicted)):
        f1, precision, recall = edge_f1(image, truth, mask)
        print(f"  {name:26s} {label:8s} PSNR {psnr(image, truth, mask):5.2f}  SSIM {ssim(image, truth, mask):.4f}  "
              f"edge F1 {f1:.3f} (P {precision:.3f} R {recall:.3f})  detail/truth {detail(image, mask) / truth_detail:.2f}", flush=True)


def main():
    net = Net().to(device())
    net.load_state_dict(torch.load(sys.argv[1], map_location=device()))
    hold = load_pair(sys.argv[2])
    sets = [("held-out south " + sys.argv[2], hold, split_south(hold["mask"])[1])]
    sets += [("unseen " + tag, load_pair(tag), None) for tag in sys.argv[3:]]
    for name, pair, mask in sets:
        mask = pair["mask"] if mask is None else mask
        rows, cols = np.where(mask)
        y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        region = mask[y0:y1, x0:x1]
        truth = pair["truth"][y0:y1, x0:x1].astype(np.float32)
        base = pair["base"][y0:y1, x0:x1].astype(np.float32)
        report(name, base, restore(net, pair["base"], pair["guide"], (y0, y1, x0, x1)), truth, region)


if __name__ == "__main__":
    main()
