import unittest
from dsin.ae.importance_map import *


class TestImportanceMap(unittest.TestCase):
    def test_channel_dim_shrinks_by1(self):
        self.info_channels = 2
        self.importance_map_layer = ImportanceMapMult(
            use_map=True, info_channels=self.info_channels
        )
        x = torch.randn([3, self.info_channels + 1, 100, 100])
        _, y = self.importance_map_layer(x)
        self.assertEqual(tuple(y.shape), (3, self.info_channels, 100, 100))

    def test_dim_of_returned_importance_map(self):
        self.info_channels = 1
        self.importance_map_layer = ImportanceMapMult(
            use_map=True, info_channels=self.info_channels
        )
        x = torch.randn([3, self.info_channels + 1, 100, 100])
        improtnace_map_mult_weights, y = self.importance_map_layer(x)
        self.assertEqual(
            tuple(improtnace_map_mult_weights.shape), (3, 1, 100, 100))

    def test_output_makes_sense(self):
        self.info_channels = 1
        self.importance_map_layer = ImportanceMapMult(
            use_map=True, info_channels=self.info_channels
        )
        x = torch.tensor(
            [
                [
                    [0, 0, 0],
                    [
                        torch.log(torch.tensor(0.5)),
                        torch.log(torch.tensor(0.5)),
                        torch.log(torch.tensor(0.5)),
                    ],
                ],
                [[3, 3, 3], [4, 4, 4]],
            ],
            dtype=torch.float32,
            requires_grad=False,
        )

        _, y = self.importance_map_layer(x)
        print(y)
        y_ = torch.tensor(
            [[[-0.3466, -0.3466, -0.3466]], [[3.8103, 3.8103, 3.8103]]],
            dtype=torch.float32,
        )
        print(y_)

        self.assertTrue(torch.allclose(y, y_, rtol=0.01))


class TestImMinMaxMap(unittest.TestCase):
    def test_only_nonnegative_clamp_at_1(self):
        mm_map = MinMaxMap()
        x = torch.arange(start=-0.5, end=1.5, step=0.1, dtype=torch.float32)
        y = MinMaxMap.apply(x)
        self.assertTrue(
            torch.all(
                y.eq(
                    torch.tensor(
                        [
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.1000,
                            0.2000,
                            0.3000,
                            0.4000,
                            0.5000,
                            0.6000,
                            0.7000,
                            0.8000,
                            0.9000,
                            1.0000,
                            1.0000,
                            1.0000,
                            1.0000,
                            1.0000,
                        ]
                    )
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
