import unittest
from dsin.ae.probclass import *
from tests import consts


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
    def test_resblock_unpadding_of_residualinput(self):
        """check maskA and maskB fit expectations."""
        l = MaskedResblock(channels=3, filter_shape=(2, 3, 3))
        x = torch.rand([2, 3, 3, 6, 6])
        y = l(x)
        # result is recevied correctly by unpad: [:, :, 2:, 2:-2, 2:-2]
        self.assertEqual(tuple(y.shape), (2, 3, 1, 2, 2))


class test_ProbClassifier(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        values_per_pixel = 6
        self.l = ProbClassifier(
            classifier_in_3d_channels=1,
            classifier_out_3d_channels=values_per_pixel,
            receptive_field=3,
        )

    def test_probclass_flow(self):
        x = torch.randn([self.batch_size, 32, 5, 5])
        y = self.l(x)
        self.assertEqual(y.shape, (2,6, 32, 5, 5))

    def test_zero_pad_layer(self):
        x = torch.ones([self.batch_size, 32, 5, 5])
        y = self.l.zero_pad_layer()(x)
        self.assertEqual(y.shape, (2, 36, 13, 13))
        self.assertTrue(torch.all(consts.y.eq(y)))


if __name__ == "__main__":
    unittest.main()
