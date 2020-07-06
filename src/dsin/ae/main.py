import torch
from torch import nn
from dsin.ae import config
from dsin.ae.si_ae import SideInformationAutoEncoder
from dsin.ae.si_net import SiNetChannelIn
from dsin.ae.loss_man import LossManager
from dsin.ae.distortions import Distortions


def main():
    si_autoencoder = SideInformationAutoEncoder(config.use_si_flag)
    loss_manager = LossManager()
    si_net_loss = loss_manager.create_si_net_loss()
    optimizer = torch.optim.Adam(si_autoencoder.parameters(), lr=1e-4)

    B = 1
    for t in range(B):
        x = torch.tensor()
        y = torch.tensor()

        (
            x_reconstructed,
            x_dec,
            x_pc,
            importance_map_mult_weights,
            x_quantizer_index_of_closest_center,
        ) = si_autoencoder(x=x, y=y)

        bit_cost_loss_value = self.lm.get_bit_cost_loss(
            pc_output=x_pc,
            quantizer_closest_center_index=x_quantizer_index_of_closest_center,
            importance_map_mult_weights=importance_map_mult_weights,
            beta_factor=config.beta,
            target_bit_cost=config.H_target,
        )
        si_net_loss_value = (
            si_net_loss(x_reconstructed, x)
            if config.use_si_flag == SiNetChannelIn.WithSideInformation
            else 0
        )
        autoencoder_loss_value = Distortions._calc_dist(
            x_dec,
            x,
            distortion=config.autoencoder_loss_distortion_to_minimize,
            cast_to_int=False,
        )
        total_loss = (
            autoencoder_loss_value * (1 - config.si_loss_weight_alpha)
            + si_net_loss_value * config.si_loss_weight_alpha
            + bit_cost_loss_value
        )

        if t % 100 == 0:
            print(t, total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()
