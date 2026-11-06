"""The restoration net and its tiled application."""
import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    """EDSR-style residual CNN without batch norm, 336 483 parameters at the default width.

    Input: 6 channels, the whole-map texture on the output grid (bicubic) and the geometry render (Detailed tiles).
    Output: 3 channels, `base + correction`: the net predicts a correction to the bicubic texture, not a new picture.
    """

    def __init__(self, channels=48, blocks=8):
        super().__init__()
        self.head = nn.Conv2d(6, channels, 3, padding=1)
        self.body = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1))
            for _ in range(blocks)
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        base = x[:, :3]
        features = self.head(x)
        for block in self.body:
            features = features + block(features) * 0.5
        return base + self.tail(features)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def restore(net, base, guide, box=None, tile=384, pad=24):
    """Run `net` over a region in tiles of `tile` px with `pad` px of context, without seams.

    base, guide: uint8 arrays (H, W, 3) on the same grid. box: (y0, y1, x0, x1), default the whole grid.
    Returns float32 (y1 - y0, x1 - x0, 3) in 0..255.
    """
    height, width = base.shape[:2]
    y0, y1, x0, x1 = box or (0, height, 0, width)
    was_training = net.training
    net.eval()
    where = next(net.parameters()).device
    out = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    for ty in range(y0, y1, tile):
        for tx in range(x0, x1, tile):
            ya, yb = max(ty - pad, 0), min(ty + tile + pad, height)
            xa, xb = max(tx - pad, 0), min(tx + tile + pad, width)
            x = np.concatenate([base[ya:yb, xa:xb], guide[ya:yb, xa:xb]], -1)
            tensor = torch.from_numpy(x).to(where).permute(2, 0, 1)[None].float() / 255
            predicted = net(tensor)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            h, w = min(ty + tile, y1) - ty, min(tx + tile, x1) - tx
            out[ty - y0:ty - y0 + h, tx - x0:tx - x0 + w] = predicted[ty - ya:ty - ya + h, tx - xa:tx - xa + w] * 255
    net.train(was_training)
    return out
