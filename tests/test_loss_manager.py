import unittest
from dsin.ae.loss_man import *
from dsin.ae import config


class TestLossMan(unittest.TestCase):
    def setUp(self):
        self.lm = LossManager()

    def test_rebase_factor(self):
        self.assertAlmostEqual(
            self.lm.log_natural_base_to_base2_factor, 1.442, delta=0.001
        )

    def test_cross_entropy_loss_in_bits_is_correct(self):
        # NCHW = 1,2,3,1
        pc_output = (
            torch.tensor([[0.1, 0.9, 0.3], [0.9, 0.1, 0.7]])
            .unsqueeze_(0)
            .unsqueeze_(-1)
        )
        quantizer_closest_center_index = (
            torch.tensor([1, 0, 1]).unsqueeze_(0).unsqueeze_(-1)
        )
        importance_map_mult_weights = torch.zeros(1, 3, 1)
        beta_factor = 1
        target_bit_cost = 0

        bit_cost = self.lm.get_bit_cost_loss(
            pc_output,
            quantizer_closest_center_index,
            importance_map_mult_weights,
            beta_factor,
            target_bit_cost,
        )
        self.assertTrue(
            torch.allclose(
                self.lm.cross_entropy_loss_in_bits.data,
                torch.tensor([[[0.3711], [0.3711], [0.5130]]]) * 1.442,
                rtol=0,
                atol=0.001,
            )
        )

        # to understand how i calculated the consts follow the next code
        # >>> loss = nn.CrossEntropyLoss(reduction='none')#'mean')
        # >>> x =  torch.tensor([[0.1 ,0.9, 0.3],[0.9, 0.1,0.7]]).unsqueeze_(0)
        # >>> x0 = [0.1 , 0.9]
        # >>> softmax_p =-np.log(np.exp(x0)/sum(np.exp(x0)))
        # >>> target_class = torch.tensor([1, 0 ,1]).unsqueeze_(0)
        # >>> print(f"{softmax_p[1]=}")
        # >>> print(f"{x.shape=}")
        # >>> print(f"{x=}")
        # >>> print(f"{loss(x,target_class)=}")
        # >>> print(f"{loss(x,target_class).shape=}")
        self.assertEqual(tuple(self.lm.cross_entropy_loss_in_bits.data.shape), (1, 3, 1))

    def test_masked_bit_entropy_is_correct(self):
        # NCHW = 1,2,3,1
        pc_output = (
            torch.tensor([[0.1, 0.9, 0.3], [0.9, 0.1, 0.7]])
            .unsqueeze_(0)
            .unsqueeze_(-1)
        )
        quantizer_closest_center_index = (
            torch.tensor([1, 0, 1]).unsqueeze_(0).unsqueeze_(-1)
        )
        # multiply each elemt in cross_entropy_loss_in_bits with 2
        importance_map_mult_weights = torch.ones(1, 3, 1) * 2
        beta_factor = 1
        target_bit_cost = 0

        bit_cost = self.lm.get_bit_cost_loss(
            pc_output,
            quantizer_closest_center_index,
            importance_map_mult_weights,
            beta_factor,
            target_bit_cost,
        )

        self.assertTrue(
            torch.allclose(
                self.lm.masked_bit_entropy.data,
                torch.tensor([[[0.3711], [0.3711], [0.5130]]]) * 1.442 * 2,
                rtol=0,
                atol=0.001,
            )
        )
        self.assertEqual(tuple(self.lm.masked_bit_entropy.data.shape), (1, 3, 1))

    def testget_bit_cost_loss_correct(self):
        # NCHW = 1,2,3,1
        pc_output = (
            torch.tensor([[0.1, 0.9, 0.3], [0.9, 0.1, 0.7]])
            .unsqueeze_(0)
            .unsqueeze_(-1)
        )
        quantizer_closest_center_index = (
            torch.tensor([1, 0, 1]).unsqueeze_(0).unsqueeze_(-1)
        )
        # multiply each elemt in cross_entropy_loss_in_bits with 3
        # so that `soft_bit_entropy` average equals 2 time the real_bit_entropy
        importance_map_mult_weights = torch.ones(1, 3, 1) * 3
        beta_factor = 1
        target_bit_cost = 0

        bit_cost = self.lm.get_bit_cost_loss(
            pc_output,
            quantizer_closest_center_index,
            importance_map_mult_weights,
            beta_factor,
            target_bit_cost,
        )

        self.assertTrue(torch.allclose(bit_cost, torch.tensor(1.2073), atol=0.001))

    def testget_bit_cost_loss_0clamp_correct(self):
        # NCHW = 1,2,3,1
        pc_output = (
            torch.tensor([[0.1, 0.9, 0.3], [0.9, 0.1, 0.7]])
            .unsqueeze_(0)
            .unsqueeze_(-1)
        )
        quantizer_closest_center_index = (
            torch.tensor([1, 0, 1]).unsqueeze_(0).unsqueeze_(-1)
        )
        # multiply each elemt in cross_entropy_loss_in_bits with 3
        # so that `soft_bit_entropy` average equals 2 time the real_bit_entropy
        importance_map_mult_weights = torch.ones(1, 3, 1) * 3
        beta_factor = 1
        target_bit_cost = 1.208

        bit_cost = self.lm.get_bit_cost_loss(
            pc_output,
            quantizer_closest_center_index,
            importance_map_mult_weights,
            beta_factor,
            target_bit_cost,
        )

        self.assertTrue(torch.allclose(bit_cost, torch.tensor(0.0), atol=0.001))


if __name__ == "__main__":
    unittest.main()
