import torch
from torch import nn
from dsin.ae import config
from dsin.ae.autoencoder_imgcomp import Encoder, Decoder
from dsin.ae.quantizer_imgcomp import Quantizer
from dsin.ae.probclass import ProbClassifier
from dsin.ae.importance_map import ImportanceMapMult
from dsin.ae.si_net import SiNet, SiNetChannelIn
from dsin.ae.si_finder import SiFinder
from dsin.ae.kitti_normalizer import ChangeImageStatsToKitti, ChangeState


class SideInformationAutoEncoder(nn.Module):
    def __init__(self,base_line_auto_enc: nn.Module):
        super().__init__()
        self.ae = base_line_auto_enc

        self.true_tuple_loss_false_just_out = True

        self.si_net = SiNet(
            in_channels=SiNetChannelIn.WithSideInformation, use_eye_init=False)

        self.si_finder = SiFinder()

        

    def forward(self, x: torch.tensor, y: torch.tensor):
        (_,  # for total loss
                x_dec,  # for auto-encoder loss
                x_pc,  # for probability classifier loss
                importance_map_mult_weights,  # for probability classifier loss
                x_quantizer_index_of_closest_center,  # for probability classifier loss
                l2_weights,
            ) = self.ae(x, y)
       
        y_syn = normalized_y_syn = None
        
        
        # N|3|H|W
        # mult y by 255 before kitti normalization
        y_syn = self.calc_y_syn(y= y * config.open_image_normalization,
                x_dec=x_dec)
        normalized_y_syn = self.ae.noramlize(y_syn)

        # N|6|H|W, concat on channel dim
        normalized_x_dec_y_syn = torch.cat((x_dec, normalized_y_syn), dim=1)

        # N|3|H|W
        x_reconstructed = self.si_net(normalized_x_dec_y_syn)
        
        self.my_tuple = (y_syn,
                normalized_y_syn,
                x_reconstructed,  # for total loss
                x_dec,  # for auto-encoder loss
                x_pc,  # for probability classifier loss
                importance_map_mult_weights,  # for probability classifier loss
                x_quantizer_index_of_closest_center,  # for probability classifier loss
                x,
                y
            )
        if self.true_tuple_loss_false_just_out:
            l2_weights = 0
            for p in self.parameters():
                l2_weights += (p ** 2).sum()
            return (x_reconstructed,  # for total loss
                x_dec,  # for auto-encoder loss
                x_pc,  # for probability classifier loss
                importance_map_mult_weights,  # for probability classifier loss
                x_quantizer_index_of_closest_center,  # for probability classifier loss
                l2_weights,
            )
       
        return x_reconstructed

    def calc_y_syn(self, y, x_dec):

        # stop gradients in si-finder and calculation of y_dec
        with torch.no_grad():
            _, y_post_map = self.ae.importance_map_layer(self.ae.enc(y))
            y_quantizer_soft, _, _ = self.ae.quantizer(y_post_map)
            y_dec = self.ae.dec(y_quantizer_soft)

            # y_syn N|3|H|W
            return self.si_finder.create_y_syn(x_dec=x_dec, y_dec=y_dec, y=y)
