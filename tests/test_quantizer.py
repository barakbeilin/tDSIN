import unittest
from dsin.ae.quantizer_imgcomp import *
from dsin.ae import config


class TestQuantizer(unittest.TestCase):
    def test_hard_and_soft_data_equals(self):
        a = Quantizer(
            num_centers=config.quantizer_num_of_centers,
            centers_initial_range=(0, 10),
            centers_regularization_factor=0.1,
            sigma=0.1,
        )

        x = torch.round(torch.rand([3, 2, 2, 2]) * 10)

        x_soft, x_hard, _ = a(x)
        self.assertTrue(torch.all(x_soft.data.eq(x_hard.data)))

    def test_non_rand_center_values(self):
        a = Quantizer(
            num_centers=3,
            centers_initial_range=(0, 10),
            centers_regularization_factor=0.1,
            sigma=0.1,
            init_centers_uniformly=True,
        )

        self.assertTrue(torch.all(a.centers.data.eq(torch.tensor([0.0, 5.0, 10.0]))))

    def test_qunatization(self):
        a = Quantizer(
            num_centers=3,
            centers_initial_range=(0, 10),
            centers_regularization_factor=0.1,
            sigma=0.1,
            init_centers_uniformly=True,
        )

        x = (
            torch.tensor([0.0, 5.0, 6.0, 10.0])
            .unsqueeze_(0)
            .unsqueeze_(0)
            .unsqueeze_(0)
        )
        x_soft, x_hard, x_index_of_closest_center = a(x)

        self.assertTrue(torch.all(x_soft.data.eq(torch.tensor([0.0, 5.0, 5.0, 10.0]))))

        self.assertTrue(torch.all(x_hard.data.eq(torch.tensor([0.0, 5.0, 5.0, 10.0]))))

        self.assertTrue(
            torch.all(x_index_of_closest_center.data.eq(torch.tensor([0, 1, 1, 2])))
        )


if __name__ == "__main__":
    unittest.main()
