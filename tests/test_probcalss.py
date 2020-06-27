import unittest
from dsin.ae.probclass import *


class test_MaskedConv3d(unittest.TestCase):
    def test_createMask(self):
        """check maskA and maskB fit expectations."""
        maskA = MaskedConv3d.create_mask((2, 3, 3), zero_center_pixel=True)
        maskB = MaskedConv3d.create_mask((2, 3, 3), zero_center_pixel=False)
        resA = torch.tensor(
            [
                [
                    [
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ]
        )
        resB = torch.tensor(
            [
                [
                    [
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    ]
                ]
            ]
        )
        self.assertTrue(torch.all(maskB.eq(resB)))
        self.assertTrue(torch.all(maskA.eq(resA)))


class test_MaskedResblock(unittest.TestCase):
    def test_createMask(self):
        """check maskA and maskB fit expectations."""
        l = MaskedResblock(channels=3, filter_shape=(2, 3, 3))
        x = torch.rand([2, 3, 3, 6, 6])
        y = l(x)
        self.assertEqual(x.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
