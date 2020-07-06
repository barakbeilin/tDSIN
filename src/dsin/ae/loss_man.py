# https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits

import torch
from torch import nn
import numpy as np


class LossManager:
    log_natural_base_to_base2_factor = np.log2(np.e)

    def __init__(self):
        # don't average over batches, will happen after importance map mult.
        self.bit_cost_loss = nn.CrossEntropyLoss(reduction="none")

    def get_bit_cost_loss(
        self,
        pc_output,
        quantizer_closest_center_index,
        importance_map_mult_weights,
        beta_factor,
        target_bit_cost,
    ):
        """
            Parameters:
                pc_output : tensor NCHW, floats
                quantizer_closest_center_index: tensor NHW, value in [0,..,C-1]
                importance_map_mult_weights: tensor NHW
                beta_factor: float
                target_bit_cost: float
            """
        # calculate crossentropy of the softmax of pc_output w.r.t the
        # indexes of the closest center of each pixel

        # bitcost :  (NCHW,NCHW) -> NHW
        self.cross_entropy_loss_in_bits = (
            self.bit_cost_loss(input=pc_output, target=quantizer_closest_center_index)
            * self.log_natural_base_to_base2_factor
        )

        mean_real_bit_entropy = torch.mean(self.cross_entropy_loss_in_bits)

        self.masked_bit_entropy = torch.mul(
            self.cross_entropy_loss_in_bits, importance_map_mult_weights
        )
        mean_masked_bit_entropy = torch.mean(self.masked_bit_entropy)

        soft_bit_entropy = 0.5 * (mean_masked_bit_entropy + mean_real_bit_entropy)

        return beta_factor * torch.max(
            soft_bit_entropy - target_bit_cost, torch.tensor(0.0, dtype=torch.float32)
        )

    @staticmethod
    def create_si_net_loss():
        return nn.L1Loss(reduction="mean")
