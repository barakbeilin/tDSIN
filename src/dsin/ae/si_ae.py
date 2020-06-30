import torch
from torch import nn
from dsin.ae import config
from dsin.ae.autoencoder_imgcomp import Encoder, Decoder
from dsin.ae.quantizer_imgcomp import Quantizer
from dsin.ae.probclass import ProbClassifier
from dsin.ae.importance_map import ImportanceMapMult
from dsin.ae.si_net import SiNet, SiNetChannelIn
from dsin.ae.si_finder import SiFinder
class SideInformationAutoencoder(nn.Module):
    def __init__(self, use_side_infomation: SiNetChannelIn,):
        super().__init__()
        self.enc = Encoder.create_module_from_const()
        self.dec = Decoder.create_module_from_const()

        self.importance_map_layer = ImportanceMapMult(use_map=True,
        info_channels=config.quantizer_num_of_channels)

        self.quantizer = Quantizer(
            num_centers=config.quantizer_num_of_centers,
            centers_initial_range=config.quantizer_center_init_range,
            centers_regularization_factor=0.1,
            sigma=0.1,
        )
        
        self.prob_classif = ProbClassifier(
            classifier_in_3d_channels=1,
            classifier_out_3d_channels=config.quantizer_num_of_centers,
            receptive_field=config.quantizer_kernel_w_h
        )

        self.si_net = SiNet(in_channels=use_side_infomation)
        self.si_finder = SiFinder()

    def forward(x: torch.tensor, y: torch.tensor, new_y: bool):
        x_hat = self.enc(x)