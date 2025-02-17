# originally generated from : dev_nb/nb__autoencoder_imgcomp.ipynb
from torch import nn
from typing import Dict, List
from dsin.ae import config
from dsin.ae.kitti_normalizer import ChangeImageStatsToKitti, ChangeState


class Enc_Cs:
    """Consts for encoder"""

    def __init__(self):

        self.in_channels_to_conv2d_1 = 3
        self.n = 128
        self.quantizer_num_of_channels = config.quantizer_num_of_channels
        self.use_heat_map = True

        self.basic_conv2d = {"padding_mode": "replicate"}
        self.modifiable_conv2d = {"conv": nn.Conv2d, **self.basic_conv2d}
        self.padding_stride2_kernel5 = {
            "kernel_size": [5, 5],
            "stride": [2, 2],
            "padding": [2, 2],
        }
        self.padding_stride1_kernel3 = {
            "kernel_size": [3, 3],
            "stride": [1, 1],
            "padding": [1, 1],
        }

        self.enc_conv2d_1 = {
            "in_channels": self.in_channels_to_conv2d_1,
            "out_channels": self.n // 2,
            **self.padding_stride2_kernel5,
            **self.modifiable_conv2d,
        }
        self.enc_conv2d_2 = {
            "in_channels": self.n // 2,
            "out_channels": self.n,
            **self.padding_stride2_kernel5,
            **self.modifiable_conv2d,
        }

        self.enc_resblock = {
            "in_channels": self.n,
            "out_channels": self.n,
            **self.padding_stride1_kernel3,
            **self.modifiable_conv2d,
        }

        self.enc_uber_resblock = {"num_of_resblocks": 3, "resblock": self.enc_resblock}
        self.enc_uber_resblocks = {
            "num_of_uberresblocks": 5,
            "uberresblock": self.enc_uber_resblock,
        }

        self.last_conv2d_out = (
            self.quantizer_num_of_channels + 1
            if self.use_heat_map
            else self.quantizer_num_of_channels
        )
        self.last_conv2d = {
            "in_channels": self.n,
            "out_channels": self.last_conv2d_out,
            **self.padding_stride2_kernel5,
            **self.basic_conv2d,
        }


class Dec_Cs(Enc_Cs):
    def __init__(self):
        super().__init__()
        # override ```sself.basic_conv2d``` since nn.ConvTranspose2d cam padd only with zeros
        self.basic_conv2d = {"padding_mode": "zeros"}
        # override ```self.modifiable_conv2d``` from nn.Conv2d to nn.ConvTranspose2d
        self.modifiable_conv2d = {"conv": nn.ConvTranspose2d, **self.basic_conv2d}
        self.padding_stride2_kernel3 = {
            "kernel_size": [3, 3],
            "stride": [2, 2],
            "padding": [1, 1],
            "output_padding": [1, 1],
        }
        self.conv2d_1 = {
            "in_channels": self.quantizer_num_of_channels,
            "out_channels": self.n,
            **self.padding_stride2_kernel3,
            **self.modifiable_conv2d,
        }

        self.dec_resblock = {
            "in_channels": self.n,
            "out_channels": self.n,
            "conv": nn.Conv2d,
            "padding_mode": "replicate",
            **self.padding_stride1_kernel3,
        }

        self.dec_uber_resblock = {"num_of_resblocks": 3, "resblock": self.dec_resblock}
        self.dec_uber_resblocks = {
            "num_of_uberresblocks": 5,
            "uberresblock": self.dec_uber_resblock,
        }

        self.dec_prelast_conv2d = {
            "in_channels": self.n,
            "out_channels": self.n // 2,
            **self.padding_stride2_kernel5,
            "output_padding": [1, 1],
            **self.modifiable_conv2d,
        }

        self.dec_last_conv2d = {
            "in_channels": self.n // 2,
            "out_channels": 3,
            **self.padding_stride2_kernel5,
            "output_padding": [1, 1],
            **self.basic_conv2d,
        }


class Encoder(nn.Module):
    def __init__(
        self,
        conv2d_1: Dict,
        conv2d_2: Dict,
        uberresblocks: Dict,
        prelast_resblock: Dict,
        last_conv2d: Dict,
    ):
        super().__init__()
        pre_res_layers = [ChangeImageStatsToKitti(direction=ChangeState.NORMALIZE)]
        # first conv layers
        pre_res_layers.extend(
            [Conv2dReluBatch2d(**conv2d_1), Conv2dReluBatch2d(**conv2d_2)]
        )

        # 5 uber blocks
        res_layers = []
        for i in range(uberresblocks["num_of_uberresblocks"]):
            res_layers.append(UberResBlock(**uberresblocks["uberresblock"]))

        # resblock after the last uber-block
        res_layers.append(ResBlock(**prelast_resblock))

        self.pre_res_model = nn.Sequential(*pre_res_layers)
        self.res_model = nn.Sequential(*res_layers)
        self.post_sum_model = nn.Conv2d(**last_conv2d)

    @classmethod
    def create_module_from_const(cls):
        enc_c = Enc_Cs()
        return cls(
            enc_c.enc_conv2d_1,
            enc_c.enc_conv2d_2,
            enc_c.enc_uber_resblocks,
            enc_c.enc_uber_resblocks["uberresblock"]["resblock"],
            enc_c.last_conv2d,
        )

    def forward(self, x):

        x_pre_res = self.pre_res_model(x)
        x_post_res = self.res_model(x_pre_res) + x_pre_res
        return self.post_sum_model(x_post_res)


class UberResBlock(nn.Module):
    def __init__(self, num_of_resblocks: int, resblock: Dict):
        super().__init__()
        layers = []
        for i in range(num_of_resblocks):
            layers.append(ResBlock(**resblock))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) + x


class ResBlock(nn.Module):
    NUM_OF_CONVS = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List,
        stride: List,
        padding_mode: str,
        padding,
        conv,
        num_of_convs: int = NUM_OF_CONVS,
    ):
        super().__init__()
        layers = []
        for i in range(num_of_convs):
            layers.append(
                conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding_mode=padding_mode,
                    padding=padding,
                )
            )
            if i == 0:
                layers.extend(
                    [
                        nn.ReLU(),
                        nn.BatchNorm2d(
                            out_channels,
                            eps=1e-03,
                            momentum=0.1,
                            affine=True,
                            track_running_stats=True,
                        ),
                    ]
                )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) + x


class Conv2dReluBatch2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List,
        stride: List,
        padding_mode: str,
        padding,
        conv,
        output_padding: List = None,
    ):
        super().__init__()
        convlution_layer = (
            conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding_mode=padding_mode,
                padding=padding,
            )
            if output_padding is None
            else conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding_mode=padding_mode,
                padding=padding,
                output_padding=output_padding,
            )
        )
        self.model = nn.Sequential(
            convlution_layer,
            nn.ReLU(),
            nn.BatchNorm2d(
                out_channels,
                eps=1e-03,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        conv2d_1: Dict,
        uberresblocks: Dict,
        post_uberblock_resblock: Dict,
        prelast_conv2d: Dict,
        last_conv2d: Dict,
    ):
        super().__init__()

        # first deconv layers
        self.pre_uberblock_model = Conv2dReluBatch2d(**conv2d_1)

        # 5 uber blocks
        layers = []
        for i in range(uberresblocks["num_of_uberresblocks"]):
            layers.append(UberResBlock(**uberresblocks["uberresblock"]))
        # resblock after the last uber-block
        layers.append(ResBlock(**post_uberblock_resblock))
        self.pre_sum_model = nn.Sequential(*layers)

        self.post_sum_model = nn.Sequential(
            Conv2dReluBatch2d(**prelast_conv2d),
            nn.ConvTranspose2d(**last_conv2d),
            ChangeImageStatsToKitti(direction=ChangeState.DENORMALIZE),
        )

    @classmethod
    def create_module_from_const(cls):
        consts = Dec_Cs()
        return cls(
            consts.conv2d_1,
            consts.dec_uber_resblocks,
            consts.dec_resblock,
            consts.dec_prelast_conv2d,
            consts.dec_last_conv2d,
        )

    def forward(self, x):
        pre_ubberblock = self.pre_uberblock_model(x)
        res_pre_conv = self.pre_sum_model(pre_ubberblock) + pre_ubberblock
        return self.post_sum_model(res_pre_conv)
