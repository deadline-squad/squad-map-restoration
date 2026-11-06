"""Decoder of BC1 (DXT1) texture data, the format of the minimap textures of the game."""
import numpy as np


def _rgb565(color):
    red = ((color >> 11) & 31).astype(np.float32) * 255 / 31
    green = ((color >> 5) & 63).astype(np.float32) * 255 / 63
    blue = (color & 31).astype(np.float32) * 255 / 31
    return np.stack([red, green, blue], -1)


def decode_bc1(raw, width, height):
    """RGB uint8 array (height, width, 3) of raw BC1 data (8 bytes per 4x4 block, rows of blocks top to bottom).

    Four-colour mode when the first endpoint is greater than the second, else three colours plus black (alpha is dropped).
    """
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(height // 4, width // 4, 8)
    c0 = blocks[..., 0].astype(np.uint16) | (blocks[..., 1].astype(np.uint16) << 8)
    c1 = blocks[..., 2].astype(np.uint16) | (blocks[..., 3].astype(np.uint16) << 8)
    p0, p1 = _rgb565(c0), _rgb565(c1)
    four_colours = (c0 > c1)[..., None]
    p2 = np.where(four_colours, (2 * p0 + p1) / 3, (p0 + p1) / 2)
    p3 = np.where(four_colours, (p0 + 2 * p1) / 3, 0.0)
    palette = np.stack([p0, p1, p2, p3], -2)
    index_bytes = blocks[..., 4:8].astype(np.uint32)
    indices = index_bytes[..., 0] | (index_bytes[..., 1] << 8) | (index_bytes[..., 2] << 16) | (index_bytes[..., 3] << 24)
    out = np.zeros((height // 4, width // 4, 4, 4, 3), np.float32)
    for row in range(4):
        for column in range(4):
            chosen = ((indices >> (2 * (row * 4 + column))) & 3).astype(np.intp)
            out[:, :, row, column, :] = np.take_along_axis(palette, chosen[..., None, None], axis=2)[:, :, 0, :]
    image = out.transpose(0, 2, 1, 3, 4).reshape(height, width, 3)
    return np.clip(image + 0.5, 0, 255).astype(np.uint8)
