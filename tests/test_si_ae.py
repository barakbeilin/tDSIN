import unittest
from dsin.ae.si_ae import *
from dsin.ae import config


class TestSIAE(unittest.TestCase):
    def test_flow_with_si(self):
        si_ae = SideInformationAutoEncoder(SiNetChannelIn.WithSideInformation)
        denoramlize = ChangeImageStatsToKitti(direction=ChangeState.DENORMALIZE)

        # change image stats to mock kitti image
        x = denoramlize(torch.randn(1, 3, 192, 144))
        y = denoramlize(torch.randn(1, 3, 192, 144))
        (
            x_reconstructed,
            x_dec,
            x_pc,
            importance_map_mult_weights,
            x_quantizer_index_of_closest_center,
        ) = si_ae(x=x, y=y)

        self.assertEqual(tuple(x_reconstructed.shape), (1, 3, 192, 144))
        self.assertEqual(tuple(x_dec.shape), (1, 3, 192, 144))
        self.assertEqual(tuple(x_pc.shape), (1, 6, 32, 192 / 8, 144 / 8))
        self.assertEqual(
            tuple(importance_map_mult_weights.shape), (1, 32, 192 / 8, 144 / 8)
        )
        self.assertEqual(
            tuple(x_quantizer_index_of_closest_center.shape), (1, 32, 192 / 8, 144 / 8)
        )

    def test_flow_without_si(self):
        si_ae = SideInformationAutoEncoder(SiNetChannelIn.NoSideInformation)
        denoramlize = ChangeImageStatsToKitti(direction=ChangeState.DENORMALIZE)
        x = denoramlize(torch.randn(1, 3, 192, 144))
        (
            x_reconstructed,
            x_dec,
            x_pc,
            importance_map_mult_weights,
            x_quantizer_index_of_closest_center,
        ) = si_ae(x=x, y=None)
        self.assertEqual(tuple(x_dec.shape), (1, 3, 192, 144))
        self.assertEqual(tuple(x_pc.shape), (1, 6, 32, 192 / 8, 144 / 8))
        self.assertEqual(
            tuple(importance_map_mult_weights.shape), (1, 32, 192 / 8, 144 / 8)
        )
        self.assertEqual(
            tuple(x_quantizer_index_of_closest_center.shape), (1, 32, 192 / 8, 144 / 8)
        )


if __name__ == "__main__":
    unittest.main()
