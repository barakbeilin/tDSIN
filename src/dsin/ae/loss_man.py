# https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits

import torch
from torch import nn
import numpy as np
from dsin.ae.distortions import Distortions
from dsin.ae.si_net import SiNetChannelIn
from dsin.ae import config


class LossManager(nn.Module):
    log_natural_base_to_base2_factor = np.log2(np.e)

    def __init__(self, use_side_infomation: SiNetChannelIn):
        super().__init__()
        # don't average over batches, will happen after importance map mult.
        self.bit_cost_loss = nn.CrossEntropyLoss(reduction="none")
        self.si_net_loss = nn.L1Loss(reduction="mean")
        self. use_side_infomation = use_side_infomation

    def forward(self, *args):
        (x_reconstructed,
         x_dec,
         x_pc,
         importance_map_mult_weights,
         x_quantizer_index_of_closest_center) = args[0]
        x_orig = args[1]

        bit_cost_loss_value = self._get_bit_cost_loss(
            pc_output=x_pc,
            quantizer_closest_center_index=x_quantizer_index_of_closest_center,
            importance_map_mult_weights=importance_map_mult_weights,
            beta_factor=config.beta,
            target_bit_cost=config.H_target,
        )

        si_net_loss_value = (
            self.si_net_loss(x_reconstructed, x_orig)
            if self.use_side_infomation == SiNetChannelIn.WithSideInformation
            else 0
        )

        autoencoder_loss_value = Distortions._calc_dist(
            x_dec,
            x_orig,
            distortion=config.autoencoder_loss_distortion_to_minimize,
            cast_to_int=False,
        )
        total_loss = (
            autoencoder_loss_value * (1 - config.si_loss_weight_alpha)
            + si_net_loss_value * config.si_loss_weight_alpha
            + bit_cost_loss_value
        )
        return total_loss

    def _get_bit_cost_loss(
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
            self.bit_cost_loss(
                input=pc_output, target=quantizer_closest_center_index)
            * self.log_natural_base_to_base2_factor
        )

        mean_real_bit_entropy = torch.mean(self.cross_entropy_loss_in_bits)

        self.masked_bit_entropy = torch.mul(
            self.cross_entropy_loss_in_bits, importance_map_mult_weights
        )
        mean_masked_bit_entropy = torch.mean(self.masked_bit_entropy)

        soft_bit_entropy = 0.5 * \
            (mean_masked_bit_entropy + mean_real_bit_entropy)

        return beta_factor * torch.max(
            soft_bit_entropy -
            target_bit_cost, torch.tensor(0.0, dtype=torch.float32)
        )
