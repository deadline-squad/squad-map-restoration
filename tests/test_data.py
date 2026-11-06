import unittest

import numpy as np
from PIL import Image

from mapsr.data import ncc, resample
from mapsr.train import centres, split_south


class Resample(unittest.TestCase):
    def test_a_box_beyond_the_image_is_clamped(self):
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        # grid of 4 px of 2 m over an image of 8 px of 1 m, shifted so the box leaves the image by half a pixel
        out = resample(image, (0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (2.0, 2.0), (4, 4), Image.BILINEAR)
        self.assertEqual(out.shape, (4, 4, 3))

    def test_the_whole_image_onto_half_the_pixels_averages_2x2(self):
        image = np.zeros((4, 4, 3), np.uint8)
        image[:2, :2] = 100
        out = resample(image, (0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (2.0, 2.0), (2, 2), Image.BOX)
        self.assertEqual(out[0, 0].tolist(), [100, 100, 100])
        self.assertEqual(out[1, 1].tolist(), [0, 0, 0])


class Correlation(unittest.TestCase):
    def test_normalised_correlation(self):
        a = np.random.default_rng(1).normal(size=(32, 32))
        self.assertAlmostEqual(ncc(a, a), 1.0, places=4)
        self.assertAlmostEqual(ncc(a, -a), -1.0, places=4)
        self.assertAlmostEqual(ncc(a, a * 5 + 3), 1.0, places=4)


class Holdout(unittest.TestCase):
    def test_south_quarter_validates_with_a_gap(self):
        mask = np.zeros((600, 100), bool)
        mask[100:500] = True
        train, validation = split_south(mask)
        cut = int(100 + 0.75 * (499 - 100))
        self.assertTrue(validation[cut:500].all() and not validation[:cut].any())
        self.assertTrue(train[100:cut - 128].all() and not train[cut - 128:].any())
        self.assertFalse((train & validation).any())

    def test_centres_keep_the_patch_inside_the_mask(self):
        mask = np.zeros((50, 50), bool)
        mask[10:40, 10:40] = True
        rows, cols = centres(mask, 5)
        self.assertEqual((rows.min(), rows.max(), cols.min(), cols.max()), (15, 34, 15, 34))


if __name__ == "__main__":
    unittest.main()
