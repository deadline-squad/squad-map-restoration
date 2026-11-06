import unittest

import numpy as np
import torch

from mapsr.model import Net, restore


class NetShape(unittest.TestCase):
    def test_parameter_count_and_output_shape(self):
        net = Net()
        self.assertEqual(sum(p.numel() for p in net.parameters()), 336483)
        self.assertEqual(net(torch.zeros(1, 6, 32, 32)).shape, (1, 3, 32, 32))

    def test_output_is_the_bicubic_input_plus_a_correction(self):
        net = Net()
        for p in net.tail.parameters():
            torch.nn.init.zeros_(p)
        x = torch.rand(1, 6, 16, 16)
        self.assertTrue(torch.allclose(net(x), x[:, :3]))


class TiledRestore(unittest.TestCase):
    def test_tiles_give_the_same_result_as_one_pass_inside_the_context(self):
        torch.manual_seed(0)
        net = Net(channels=8, blocks=2)
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (80, 90, 3), dtype=np.uint8)
        guide = rng.integers(0, 255, (80, 90, 3), dtype=np.uint8)
        whole = restore(net, base, guide, tile=1000, pad=0)
        tiled = restore(net, base, guide, tile=32, pad=8)  # receptive field of 2 blocks + head + tail = 6 px
        self.assertEqual(tiled.shape, (80, 90, 3))
        self.assertLess(np.abs(whole - tiled).max(), 1e-2 * 255)

    def test_a_box_restores_only_that_region(self):
        net = Net(channels=8, blocks=2)
        base = np.zeros((40, 40, 3), np.uint8)
        out = restore(net, base, base, box=(10, 30, 5, 25), tile=16, pad=8)
        self.assertEqual(out.shape, (20, 20, 3))


if __name__ == "__main__":
    unittest.main()
