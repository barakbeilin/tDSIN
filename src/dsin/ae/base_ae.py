import torch
from torch import nn
from dsin.ae import config
from dsin.ae.autoencoder_imgcomp import Encoder, Decoder
from dsin.ae.quantizer_imgcomp import Quantizer
from dsin.ae.probclass import ProbClassifier
from dsin.ae.importance_map import ImportanceMapMult
from dsin.ae.si_finder import SiFinder
from dsin.ae.kitti_normalizer import ChangeImageStatsToKitti, ChangeState
from dsin.ae.data_manager.data_loader import ImageSiTuple

class BaseAutoEncoder(nn.Module):
    def __init__(self):
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

        self.noramlize = ChangeImageStatsToKitti(
            direction=ChangeState.NORMALIZE)
        self.denoramlize = ChangeImageStatsToKitti(
            direction=ChangeState.DENORMALIZE)

    # def forward(self, x: torch.tensor, y: torch.tensor):
    def forward(self, combined_img_si_img):
        x , y = ImageSiTuple.data_to_si_img_and_img(combined_img_si_img)
        # N| nof inpput Quantization Channels + 1|H/8|W/8
        # TODO: DELETE AND PASS INTO importance_map_layer DIRECTLY
        # x is in [0,1] color range so need to multiply it back to [0,255] range
        # for normalization to work.
        x_enc = self.enc(x * config.open_image_normalization)

        # improtance map - N|nof input quantization channels|H/8|W/8
        # x_post_map - N|nof input quantization channels|H/8|W/8
        importance_map_mult_weights, x_post_map = self.importance_map_layer(
            x_enc)

        # N|nof input Quntization Channels|H|W
        # x_soft - data values are from centers alphabet, gradients from softmax
        # (2) x_hard - not used, include data of x_soft without gardients.
        # x_index_of_closest_center- indexes of the center values inside x_soft.data
        x_quantizer_soft, _, x_quantizer_index_of_closest_center = self.quantizer(
            x_post_map
        )

        # create a copy without gradient to prevent multiple gradients flowing
        # into the importance map
        # x_pc - -5dims of size:
        #   N|nof Quantization Centers|nof input quantization channels |H|W|
        x_pc = self.prob_classif(x_quantizer_soft.detach())

        # N|3|H|W
        x_dec = self.dec(x_quantizer_soft)
      
        
        l2_weights = None 
        for name, param in self.named_parameters():
            
            if name.startswith('enc') or name.startswith('dec'):
                if l2_weights is None:
                    l2_weights = 0.5 * torch.sum(param**2)
                else:
                    l2_weights = l2_weights + 0.5 * param.norm(2)**2

  
        
        self.my_tuple = (None, None,None,
                x_dec,  # for auto-encoder loss
                x_pc,  # for probability classifier loss
                importance_map_mult_weights,  # for probability classifier loss
                x_quantizer_index_of_closest_center,  # for probability classifier loss
                x,
                y,
                l2_weights
            )

         




       
       
        # x_dec is after clamping to level between 0 and 1 by sigmoid
        return x_dec
