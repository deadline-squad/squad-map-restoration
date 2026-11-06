import unittest

import numpy as np

from mapsr.bc1 import decode_bc1


class DecodeBc1(unittest.TestCase):
    def block(self, c0, c1, indices):
        return bytes([c0 & 255, c0 >> 8, c1 & 255, c1 >> 8]) + indices.to_bytes(4, "little")

    def test_four_colour_block(self):
        # c0 pure red, c1 pure blue; first row of indices 0, 1, 2, 3, every other pixel index 0
        raw = self.block(0xF800, 0x001F, 0b11100100)
        image = decode_bc1(raw, 4, 4)
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(image[0].tolist(), [[255, 0, 0], [0, 0, 255], [170, 0, 85], [85, 0, 170]])
        self.assertTrue((image[1:] == [255, 0, 0]).all())

    def test_three_colour_block_has_midpoint_and_black(self):
        # c0 < c1: index 2 is the midpoint, index 3 black
        raw = self.block(0x001F, 0xF800, 0b11100100)
        image = decode_bc1(raw, 4, 4)
        self.assertEqual(image[0].tolist(), [[0, 0, 255], [255, 0, 0], [128, 0, 128], [0, 0, 0]])

    def test_blocks_are_laid_out_row_by_row(self):
        red, blue = self.block(0xF800, 0xF800, 0), self.block(0x001F, 0x001F, 0)
        image = decode_bc1(red + blue + blue + red, 8, 8)
        self.assertEqual(image[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(image[0, 4].tolist(), [0, 0, 255])
        self.assertEqual(image[4, 0].tolist(), [0, 0, 255])
        self.assertEqual(image[4, 4].tolist(), [255, 0, 0])


if __name__ == "__main__":
    unittest.main()
