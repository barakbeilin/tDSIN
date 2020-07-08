from dsin.ae.distortions import DistTypes
from dsin.ae.si_net import SiNetChannelIn

# SiNetChannelIn.WithSideInformation
use_si_flag = SiNetChannelIn.NoSideInformation

H_target = 2 * 0.02  # == 64/C * bpp
beta = 500
quantizer_num_of_channels = 32
quantizer_num_of_centers = 6
quantizer_center_init_range = (-2, 2)
quantizer_kernel_w_h = 3

autoencoder_loss_distortion_to_minimize = DistTypes.MAE
si_loss_weight_alpha = 0.7

open_image_normalization = 255.0
