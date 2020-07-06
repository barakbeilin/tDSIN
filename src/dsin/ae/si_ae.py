import torch
from torch import nn
from dsin.ae import config
from dsin.ae.autoencoder_imgcomp import Encoder, Decoder
from dsin.ae.quantizer_imgcomp import Quantizer
from dsin.ae.probclass import ProbClassifier
from dsin.ae.importance_map import ImportanceMapMult
from dsin.ae.si_net import SiNet, SiNetChannelIn
from dsin.ae.si_finder import SiFinder
from dsin.ae.kitti_normalizer import ChangeImageStatsToKitti, ChangeStates


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

        self.noramlize = ChangeImageStatsToKitti(direction=ChangeStates.NORMALIZE)
        self.denoramlize = ChangeImageStatsToKitti(direction=ChangeStates.DENORMALIZE)

    def forward(
        self, x: torch.tensor, y: torch.tensor, new_y: bool, train_only_ae: bool
    ):

        # N| nof inpput Quantization Channels + 1|H|W
        # TODO: DELETE AND PASS INTO importance_map_layer DIRECTLY
        x_enc = self.enc(x)

        # improtance map - N1HW
        # x_post_map - N|nof input quantization channels|H|W
        importance_map_mult_weights, x_post_map = self.importance_map_layer(x_enc)

        # x_quantizer(a.k.a z-hat) - N|nof input Quntization Channels|H|W
        x_quantizer = self.quantizer(x_post_map)

        # create a copy without gradient to prevent multiple gradients flowing
        # into the importance map
        # x_pc - -5dims of size:
        #   N|nof Quantization Centers|nof input quantization channels |H|W|
        x_pc = self.prob_classif(x_quantizer.detach())

        # N|3|H|W
        x_dec = self.dec(x_quantizer)
        normalized_x_dec = self.noramlize(x_dec)

        # N|3|H|W
        # TODO: DELETE AND PASS INTO cat DIRECTLY
        normalized_y_syn = self.noramlize(self.calc_y_syn(y=y, x_dec=normalized_x_dec))

        # N|6|H|W, concat on channel dim
        # TODO: DELETE AND PASS INTO SI_NET DIRECTLY
        normalized_x_dec_y_syn = torch.cat((normalized_x_dec, normalized_y_syn), dim=1)

        # N|3|H|W
        x_reconstructed = self.denoramlize(self.si_net(normalized_x_dec_y_syn))
        return x_reconstructed, x_dec, x_pc

    def calc_y_syn(self, y, x_dec):
        # stop gradients in si-finder and calculation of y_dec
        with torch.no_grad():
            _, y_post_map = self.importance_map_layer(self.enc(y))
            y_dec = self.dec(self.quantizer(y_post_map))

            # y_syn N|3|H|W
            return self.si_finder.create_y_syn(x_dec=x_dec, y_dec=y_dec)
