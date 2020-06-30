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

        x_soft, x_hard, x_index_of_center = a(x)
        self.assertTrue(torch.all(x_soft.data.eq(x_hard.data)))


if __name__ == "__main__":
    unittest.main()
