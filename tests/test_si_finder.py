import unittest
from unittest.mock import patch
from dsin.ae.si_finder import *
from tests import consts


class TestSiFinder(unittest.TestCase):
    @patch("dsin.ae.si_finder.SiFinder.KERNEL_SIZE", 3)
    def test_ysyn_creation(self):
        sf = SiFinder()
        y = torch.tensor(
            [
                [
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [2, 2, 2, 3, 3, 3],
                    [2, 2, 2, 3, 3, 3],
                    [2, 2, 2, 3, 3, 3],
                ],
                [
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [2, 2, 2, 3, 3, 3],
                    [2, 2, 2, 3, 3, 3],
                    [2, 2, 2, 3, 3, 3],
                ],
                [
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [2, 2, 2, 3, 3, 3],
                    [2, 2, 2, 3, 3, 3],
                    [2, 2, 2, 3, 3, 3],
                ],
            ],
            dtype=torch.float32,
        ).unsqueeze_(0)

        x = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.float32,
        ).unsqueeze_(0)

        # y_dec = x.clone().detach()
        y_dec = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.float32,
        ).unsqueeze_(0)

        y_syn = sf.create_y_syn(x_dec=x, y_dec=y_dec, y=y)
        self.assertTrue(
            torch.all(
                torch.eq(
                    y_syn,
                    torch.tensor(
                        [
                            [
                                [
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                ],
                                [
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                ],
                                [
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                ],
                            ]
                        ]
                    ),
                )
            )
        )

    @patch("dsin.ae.si_finder.SiFinder.KERNEL_SIZE", 3)
    def test_best_patch_is_returned(self):
        sf = SiFinder()
        x = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.float32,
        ).unsqueeze_(0)

        y = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.float32,
        ).unsqueeze_(0)

        self.assertEqual(x.shape, (1, 3, 6, 6))
        y_dec_patch_indexs = sf._get_best_patch_index(x, y)

        self.assertEqual(
            y_dec_patch_indexs,
            (
                (torch.tensor(0), torch.tensor(0)),
                (torch.tensor(0), torch.tensor(3)),
                (torch.tensor(3), torch.tensor(0)),
                (torch.tensor(3), torch.tensor(3)),
            ),
        )

    @patch("dsin.ae.si_finder.SiFinder.KERNEL_SIZE", 8)
    def test_create_x_patches(self):
        sf = SiFinder()
        x = (
            torch.stack([torch.eye(n=8), torch.eye(n=8) * 2, torch.eye(n=8) * 3])
            .repeat(1, 2, 2)
            .unsqueeze_(0)
        )
        self.assertEqual(tuple(x.shape), (1, 3, 16, 16))

        x_patches = sf._get_x_patches(x)
        self.assertEqual(tuple(x_patches.shape), (4, 3, 8, 8))

        # loop through all patches
        for i in range(x_patches.shape[0]):
            self.assertTrue(
                torch.all(
                    torch.eq(
                        x_patches[i, :, :, :],
                        torch.tensor(
                            [
                                [
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                    ],
                                    [
                                        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                                    ],
                                    [
                                        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                                    ],
                                ]
                            ]
                        ),
                    )
                )
            )

    @patch("dsin.ae.si_finder.SiFinder.KERNEL_SIZE", 8)
    def test_pearson_corr(self):
        sf = SiFinder()
        x = (
            torch.stack([torch.eye(n=8), torch.eye(n=8) * 2, torch.eye(n=8) * 3])
            .repeat(1, 2, 2)
            .unsqueeze_(0)
        )
        self.assertEqual(tuple(x.shape), (1, 3, 16, 16))

        x_patches = sf._get_x_patches(x)
        out = SiFinder.pearson_corr(x_patches, x)
        self.assertEqual(tuple(out.shape), (1, 4, 9, 9))

        self.assertTrue(
            torch.allclose(
                out,
                torch.tensor(
                    [
                        [
                            [
                                [
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                ],
                                [
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                ],
                                [
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                ],
                            ],
                            [
                                [
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                ],
                                [
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                ],
                                [
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                ],
                            ],
                            [
                                [
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                ],
                                [
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                ],
                                [
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                ],
                            ],
                            [
                                [
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                ],
                                [
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                ],
                                [
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                    -0.1325,
                                ],
                                [
                                    1.1038,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    -0.1325,
                                    1.1038,
                                ],
                            ],
                        ]
                    ]
                ),
                rtol=0.001,
            )
        )


if __name__ == "__main__":
    unittest.main()
