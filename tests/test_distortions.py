import unittest
from dsin.ae.distortions import *


class TestDistortions(unittest.TestCase):
    def test_all_dists_when_minimizing_MAE(self):
        x = torch.ones([2, 3, 10, 10], dtype=torch.float32)
        y = x + 1

        d = Distortions(x, y, DistTypes.MAE, is_training=True)

        self.assertTrue(
            torch.allclose(d.get_distortion(DistTypes.MAE), torch.tensor(1.0))
        )

        self.assertTrue(
            torch.allclose(d.get_distortion(DistTypes.MSE), torch.tensor(1.0))
        )

        self.assertTrue(
            torch.allclose(d.get_distortion(DistTypes.PSNR), torch.tensor(51.8692))
        )

    def test_all_MSSIM_when_minimizing_MSSIM(self):
        x = torch.ones([2, 3, 256, 256], dtype=torch.float32)
        y = x

        d = Distortions(x, y, DistTypes.MS_SSMIM, is_training=True)

        # check perfect match
        self.assertTrue(
            torch.allclose(d.get_distortion(DistTypes.MS_SSMIM), torch.tensor(0.0))
        )

        y = x + 1
        # check non-perfect match , I just put some number in the result to monitor
        # if further development might change it (then test will fail)
        d2 = Distortions(x, y, DistTypes.MS_SSMIM, is_training=True)
        self.assertTrue(
            torch.allclose(d2.get_distortion(DistTypes.MS_SSMIM), torch.tensor(60.2528))
        )


if __name__ == "__main__":
    unittest.main()
