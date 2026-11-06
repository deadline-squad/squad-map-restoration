"""Image metrics over a boolean mask: PSNR, SSIM, edge F1 and the amount of fine detail."""
import numpy as np
from scipy import ndimage


def psnr(a, b, mask):
    """PSNR (dB) of two RGB images over the masked pixels (inf when they are equal there)."""
    mse = float(((a - b)[mask] ** 2).mean())
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def ssim(a, b, mask):
    """Mean SSIM of the luminance (7 x 7 uniform window) over the masked pixels."""
    a, b = a.mean(-1), b.mean(-1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mean = lambda v: ndimage.uniform_filter(v, 7)
    ma, mb = mean(a), mean(b)
    va, vb, cov = mean(a * a) - ma * ma, mean(b * b) - mb * mb, mean(a * b) - ma * mb
    score = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(score[mask].mean())


def edge_map(rgb, percentile=90):
    """Boolean edge map: the strongest `100 - percentile` % of the gradient magnitude of the smoothed luminance."""
    smooth = ndimage.gaussian_filter(rgb.mean(-1), 0.8)
    magnitude = np.hypot(ndimage.sobel(smooth, 1), ndimage.sobel(smooth, 0))
    return magnitude > np.percentile(magnitude, percentile)


def edge_f1(predicted, truth, mask, tolerance=1):
    """(F1, precision, recall) of the predicted edges against the truth's, with `tolerance` px of slack."""
    pe, te = edge_map(predicted), edge_map(truth)
    pe_near = ndimage.binary_dilation(pe, iterations=tolerance)
    te_near = ndimage.binary_dilation(te, iterations=tolerance)
    precision = (pe & te_near & mask).sum() / max((pe & mask).sum(), 1)
    recall = (te & pe_near & mask).sum() / max((te & mask).sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return float(f1), float(precision), float(recall)


def detail(rgb, mask):
    """Mean |Laplacian| of the luminance over the mask: the amount of fine detail."""
    return float(np.abs(ndimage.laplace(rgb.mean(-1)))[mask].mean())
