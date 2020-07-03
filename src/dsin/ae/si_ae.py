import torch
from torch import nn
from dsin.ae import config
from dsin.ae.autoencoder_imgcomp import Encoder, Decoder
from dsin.ae.quantizer_imgcomp import Quantizer
from dsin.ae.probclass import ProbClassifier
from dsin.ae.importance_map import ImportanceMapMult
from dsin.ae.si_net import SiNet, SiNetChannelIn
from dsin.ae.si_finder import SiFinder


class SideInformationEncoder(nn.Module):
    def __init__(self, use_side_infomation: SiNetChannelIn):
        super().__init__()

        self.enc = Encoder.create_module_from_const()
        self.dec = Decoder.create_module_from_const()

        self.importance_map_layer = ImportanceMapMult(
            use_map=True, info_channels=config.quantizer_num_of_channels
        )

        self.quantizer = Quantizer(
            num_centers=config.quantizer_num_of_centers,
            centers_initial_range=config.quantizer_center_init_range,
            centers_regularization_factor=0.1,
            sigma=0.1,
            init_centers_uniformly=True,
        )

        self.prob_classif = ProbClassifier(
            classifier_in_3d_channels=1,
            classifier_out_3d_channels=config.quantizer_num_of_centers,
            receptive_field=config.quantizer_kernel_w_h,
        )

        self.si_net = SiNet(in_channels=use_side_infomation, use_eye_init=True)
        self.si_finder = SiFinder()

    def forward(
        self, x: torch.tensor, y: torch.tensor, new_y: bool, train_only_ae: bool
    ):
        self.x_enc = self.enc(x)
        self.importance_map_mult_weights, self.x_post_map = self.importance_map_layer(
            self.x_enc
        )

        # z-hat
        self.x_quantizer = self.quantizer(self.x_post_map)

        # create a copy without gradient to prevent multiple gradients flowing
        # into the importance map
        self.x_pc = self.prob_classif(self.x_quantizer.detach())

        self.x_dec = self.dec(self.x_quantizer)
        self.y_syn = self.si_finder(self.x_dec, y, new_y)
        self.x_reconstructed = slef.si_net(# how to concataane ?)

