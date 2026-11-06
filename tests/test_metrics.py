import unittest

import numpy as np

from mapsr.metrics import detail, edge_f1, psnr, ssim


def picture():
    rng = np.random.default_rng(0)
    image = np.zeros((64, 64, 3), np.float32)
    image[16:48, 16:48] = 200
    return image + rng.normal(0, 2, image.shape).astype(np.float32)


class Metrics(unittest.TestCase):
    def test_psnr_of_a_constant_offset(self):
        a = np.full((8, 8, 3), 100.0)
        mask = np.ones((8, 8), bool)
        self.assertAlmostEqual(psnr(a + 10, a, mask), 20 * np.log10(255 / 10), places=6)

    def test_identical_images_are_perfect(self):
        image, mask = picture(), np.ones((64, 64), bool)
        self.assertAlmostEqual(ssim(image, image, mask), 1.0, places=6)
        self.assertAlmostEqual(edge_f1(image, image, mask)[0], 1.0, places=6)

    def test_blur_loses_edges_and_detail(self):
        from scipy import ndimage
        image, mask = picture(), np.ones((64, 64), bool)
        blurred = ndimage.gaussian_filter(image, (3, 3, 0))
        self.assertLess(ssim(blurred, image, mask), 0.99)
        self.assertLess(detail(blurred, mask), detail(image, mask))

    def test_only_the_masked_pixels_count(self):
        a = np.zeros((8, 8, 3))
        b = a.copy()
        b[:, 4:] = 50
        left = np.zeros((8, 8), bool)
        left[:, :4] = True
        self.assertEqual(psnr(a, b, left), float("inf"))
        self.assertAlmostEqual(psnr(a, b, ~left), 20 * np.log10(255 / 50), places=6)


if __name__ == "__main__":
    unittest.main()
