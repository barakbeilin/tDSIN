import unittest
import torch
from dsin.ae.kitti_normalizer import ChangeState, ChangeImageStatsToKitti
from tests import consts


class TestNormalization(unittest.TestCase):
    def test_norm_flow_6channels(self):
        INPUT_CHANNELS = 6
        normalize = ChangeImageStatsToKitti(
            ChangeState.NORMALIZE, input_channels=INPUT_CHANNELS
        )
        denormalize = ChangeImageStatsToKitti(
            ChangeState.DENORMALIZE, input_channels=INPUT_CHANNELS
        )

        x = torch.randn([2, INPUT_CHANNELS, 50, 50])

        self.assertAlmostEqual(torch.mean(x).data, 0.0, delta=0.2)
        self.assertAlmostEqual(torch.var(x).data, 1.0, delta=0.3)

        denorm = denormalize(x)
        self.assertTrue(tuple(denorm.shape) == (2, INPUT_CHANNELS, 50, 50))
        self.assertTrue(
            torch.allclose(
                torch.mean(denorm, dim=(0, 2, 3)),
                torch.tensor(
                    [
                        93.70454143384742,
                        98.28243432206516,
                        94.84678088809876,
                        93.70454143384742,
                        98.28243432206516,
                        94.84678088809876,
                    ],  # repeat twice the kitti mean stats
                    dtype=torch.float32,
                ),
                atol=30,
                rtol=0,
            )
        )

        self.assertTrue(
            torch.allclose(
                torch.var(denorm, dim=(0, 2, 3)),
                torch.tensor(
                    [
                        5411.79935676,
                        5758.60456747,
                        5890.31451232,
                        5411.79935676,
                        5758.60456747,
                        5890.31451232,
                    ],  # repeat twice the kitti var stats
                    dtype=torch.float32,
                ),
                atol=0,
                rtol=0.5,
            )
        )

        normalized = normalize(denorm)

        self.assertTrue(
            torch.allclose(
                torch.mean(normalized, dim=(0, 2, 3)),
                torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32,),
                atol=0.3,
                rtol=0,
            )
        )

        self.assertTrue(
            torch.allclose(
                torch.var(normalized, dim=(0, 2, 3)),
                torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32,),
                atol=0.3,
                rtol=0,
            )
        )

        self.assertTrue(tuple(normalized.shape) == (2, INPUT_CHANNELS, 50, 50))

    def test_norm_flow_3channels(self):
        INPUT_CHANNELS = 3
        normalize = ChangeImageStatsToKitti(ChangeState.NORMALIZE)
        denormalize = ChangeImageStatsToKitti(ChangeState.DENORMALIZE)

        x = torch.randn([2, INPUT_CHANNELS, 50, 50])

        self.assertAlmostEqual(torch.mean(x).data, 0.0, delta=0.2)
        self.assertAlmostEqual(torch.var(x).data, 1.0, delta=0.3)

        denorm = denormalize(x)
        self.assertTrue(tuple(denorm.shape) == (2, INPUT_CHANNELS, 50, 50))
        self.assertTrue(
            torch.allclose(
                torch.mean(denorm, dim=(0, 2, 3)),
                torch.tensor(
                    [93.70454143384742, 98.28243432206516, 94.84678088809876],
                    dtype=torch.float32,
                ),
                atol=30,
                rtol=0,
            )
        )

        self.assertTrue(
            torch.allclose(
                torch.var(denorm, dim=(0, 2, 3)),
                torch.tensor(
                    [5411.79935676, 5758.60456747, 5890.31451232], dtype=torch.float32,
                ),
                atol=0,
                rtol=0.5,
            )
        )

        normalized = normalize(denorm)

        self.assertTrue(
            torch.allclose(
                torch.mean(normalized, dim=(0, 2, 3)),
                torch.tensor([0, 0, 0], dtype=torch.float32,),
                atol=0.3,
                rtol=0,
            )
        )

        self.assertTrue(
            torch.allclose(
                torch.var(normalized, dim=(0, 2, 3)),
                torch.tensor([1, 1, 1], dtype=torch.float32,),
                atol=0.3,
                rtol=0,
            )
        )

        self.assertTrue(tuple(normalized.shape) == (2, INPUT_CHANNELS, 50, 50))


if __name__ == "__main__":
    unittest.main()
