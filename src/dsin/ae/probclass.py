# originally generated from: dev_nb/nb__probclass.ipynb

import torch
import torch.nn.functional as F
from torch import nn
from enum import Enum, auto
from typing import Tuple


class MaskedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_mask):
        """
        custom config of conv3d:
        - use VALID padding (i.e no padding)
        - weight init - xaviel_init
        - bias init - zero_init
        """
        super().__init__()
        self.kernel_size = tuple(filter_mask.shape)[2:]  # NDCHW -> CHW
        self.register_buffer("filter_mask", filter_mask)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            bias=True,
        )

        # initalize the conv layer
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        self._mask_conv_filter()
        return self.conv(x)

    def _mask_conv_filter(self):
        with torch.no_grad():
            # print((self.conv.weight.shape))
            # print((self.filter_mask.shape))
            # import ipdb; ipdb.set_trace()
            self.conv.weight = nn.Parameter(self.conv.weight * self.filter_mask)

    @staticmethod
    def create_mask(filter_shape: Tuple, zero_center_pixel: bool):
        """create 5d mask that includes all pixel's in strictly before
        Parameters : 
        filter_shape: Tuple of the shape, should be 3d
        zero_center_pixel: maskA<->True maskB<->False
        """
        assert (
            len(filter_shape) == 3
        ), f"filter_shape size must be 3 instead {len(filter_shape.size)}"

        K = filter_shape[2]  # K = W
        # mask is DHW
        mask = torch.ones(filter_shape, dtype=torch.float32, requires_grad=False)

        # zero out D=1,
        if zero_center_pixel:
            # zero out- everything to the right of the central pixel,
            # including the central pixel
            mask[-1, K // 2, K // 2 :] = 0
        else:
            # zero out- everything to the right of the central pixel,
            # not including the central pixel
            mask[-1, K // 2, K // 2 + 1 :] = 0

        # - all rows below the central row
        mask[-1, K // 2 + 1 :, :] = 0

        # Make into ioDHW, for broadcasting with 3D filters
        # !!! This is different than tensorflow since the dimension order is
        # different between tf and torch, notice the the (in)out-channel dimensions
        # are at the end for TF while they are in the begininng for torch.
        # conv3d.weight dimensions are:
        # TF :
        # [filter_depth, filter_height, filter_width, in_channels,out_channels]
        #   Link:
        #   https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/conv3d
        #
        # torch :
        # (out_channels, in_channels, kernel_size[0],kernel_size[1],kernel_size[2])
        #   Link:
        #   https://pytorch.org/docs/master/generated/torch.nn.Conv3d.html

        # TF: mask.unsqueeze_(-1).unsqueeze_(-1)
        # oiDHW ->outChannels|inChannels|Depth|H|W
        mask.unsqueeze_(0).unsqueeze_(0)
        return mask


class MaskedResblock(nn.Module):
    def __init__(self, channels, filter_shape):
        super().__init__()

        conv0 = self._create_mask_b_conv(channels, filter_shape)
        conv2 = self._create_mask_b_conv(channels, filter_shape)

        self.model = nn.Sequential(conv0, nn.ReLU(), conv2)

    def forward(self, x):
        y = self.model(x)
        unpad_x = x[:, :, 2:, 2:-2, 2:-2]  # fit residual input padding
        # unpad_x = x[..., 2:, 2:-2, 2:-2, :]  # original
        assert y.shape == unpad_x.shape
        # >>help in case of bug
        # print('x\unpad_x\conv_output')
        # print(x.shape)
        # print(unpad_x.shape)
        # print(y.shape)

        return unpad_x + y

    @staticmethod
    def _create_mask_b_conv(channels, filter_shape):
        maskB = MaskedConv3d.create_mask(filter_shape, zero_center_pixel=False)
        return MaskedConv3d(
            in_channels=channels, out_channels=channels, filter_mask=maskB,
        )


class ProbClassifier(nn.Module):
    def __init__(
        self, classifier_in_3d_channels, classifier_out_3d_channels, receptive_field=3
    ):
        super().__init__()

        self.receptive_field = receptive_field
        K = receptive_field
        self.filter_shape = (K // 2 + 1, K, K)  # CHW
        CONV0_OUT_CH = 24

        mask_A = MaskedConv3d.create_mask(self.filter_shape, zero_center_pixel=True)
        conv0 = MaskedConv3d(
            in_channels=classifier_in_3d_channels,
            out_channels=CONV0_OUT_CH,
            filter_mask=mask_A,
        )

        resblock = MaskedResblock(channels=CONV0_OUT_CH, filter_shape=self.filter_shape)

        mask_B = MaskedConv3d.create_mask(self.filter_shape, zero_center_pixel=False)
        conv2 = MaskedConv3d(
            in_channels=CONV0_OUT_CH,
            out_channels=classifier_out_3d_channels,
            filter_mask=mask_B,
        )
        self.premodel = self.zero_pad_layer()
        self.model = nn.Sequential(conv0, nn.ReLU(), resblock, conv2, nn.ReLU())

    def forward(self, x):
        # import ipdb ;        ipdb.set_trace()
        assert len(x.shape) == 4  # NCHW
        padded_x = self.premodel(x)
        padded_x.unsqueeze_(1)  # NTCHW T->channel dim of the 3dconv

        assert padded_x.shape[1] == 1  # T = 1

        return self.model(padded_x)

    def zero_pad_layer(self):
        """
        :param x: NCHW tensorflow Tensor or numpy array
        """
        nof_conv_layers_classifier = 4  # 4 convd3d layers
        context_size = nof_conv_layers_classifier * (self.receptive_field - 1) + 1

        pad = context_size // 2  # 4

        # padding_left , padding_right , padding_top , padding_bottom , padding_front , padding_back

        pad_N = (0, 0)
        pad_C = (pad, 0)
        pad_HW = (pad, pad)

        pads = pad_HW + pad_HW + pad_C + pad_N
        # import ipdb; ipdb.set_trace()
        return nn.ConstantPad3d(pads, value=0)
