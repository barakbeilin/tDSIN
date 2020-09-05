import torch
from torch import nn
from enum import Enum
from dsin.ae.kitti_normalizer import ChangeImageStatsToKitti, ChangeState


class SiNetChannelIn(Enum):
    WithSideInformation = 6
    NoSideInformation = 3



class Conv2dDSIN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_mode="replicate",
                padding=[dilation, dilation],
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
        ]
        self._weight_init()
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

    def _weight_init(self):
        # kaiming uniform init
        for layer in self.layers:
            if type(layer) is nn.Conv2d:
                weight = getattr(layer, "weight")
                nn.init.kaiming_uniform_(
                    weight,
                    a=self.negative_slope,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )


class DilatedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.layers = [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_mode="replicate",
            padding=[dilation, dilation],
        ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(out_channels,
                           eps=1e-03,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True,
                           ),
        ]
        self._weight_init()
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x) + x

    def _weight_init(self):
        # kaiming uniform init
        for layer in self.layers:
            if type(layer) is nn.Conv2d:
                weight = getattr(layer, "weight")
                nn.init.kaiming_uniform_(
                    weight,
                    a=self.negative_slope,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )


class SiNetDSIN(nn.Module):
    NOF_INTERNAL_LAYERS = 7
    NEG_SLOPE = 0.2

    def __init__(self, in_channels: SiNetChannelIn):
        super().__init__()

        internal_layers = [
            Conv2dDSIN(
                in_channels=32,
                out_channels=32,
                kernel_size=[3, 3],
                dilation=2 ** (i + 1),
                negative_slope=self.NEG_SLOPE)
            for i in range(self.NOF_INTERNAL_LAYERS)]

        pre_layers = [ 
            Conv2dDSIN(
                in_channels=in_channels.value,
                out_channels=32,
                kernel_size=[3, 3],
                dilation=1,
                negative_slope=self.NEG_SLOPE),
         
           ]

        post_layers = [
            Conv2dDSIN(
                in_channels=32,
                out_channels=32,
                kernel_size=[3, 3],
                dilation=1,
                negative_slope=self.NEG_SLOPE),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=[1, 1],),
            ChangeImageStatsToKitti(direction=ChangeState.DENORMALIZE),
        ]

        # self.layers = pre_layers + post_layers + internal_layers

        self.pre_model = nn.Sequential(*pre_layers)
        self.internal_model = nn.Sequential(*internal_layers)
        self.post_model = nn.Sequential(*post_layers)
        self._weight_init()

    def _weight_init(self):
        # kaiming uniform init
        for layer in self.modules():
            if type(layer) is nn.Conv2d:
                weight = getattr(layer, "weight")
                nn.init.kaiming_uniform_(
                    weight,
                    a=self.NEG_SLOPE,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )

    def forward(self, x):
        return self.post_model(self.internal_model(self.pre_model(x)))


class SiNet(nn.Module):
    NOF_INTERNAL_LAYERS = 7
    NEG_SLOPE = 0.2

    def __init__(self, in_channels: SiNetChannelIn):
        super().__init__()

        internal_layers = [
            DilatedResBlock(
                in_channels=32,
                out_channels=32,
                kernel_size=[3, 3],
                dilation=2 ** (i + 1),
                negative_slope=self.NEG_SLOPE)
            for i in range(self.NOF_INTERNAL_LAYERS)]

        pre_layers = [
            nn.Conv2d(
                in_channels=in_channels.value,
                out_channels=32,
                kernel_size=[3, 3],
                padding_mode="replicate",
                padding=[1, 1],
            ),
            nn.LeakyReLU(negative_slope=self.NEG_SLOPE),
            nn.BatchNorm2d(32,
                           eps=1e-03,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True,
                           ),
        ]

        post_layers = [
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=[1, 1],),
            ChangeImageStatsToKitti(direction=ChangeState.DENORMALIZE),
        ]

        # self.layers = pre_layers + post_layers + internal_layers

        self.pre_model = nn.Sequential(*pre_layers)
        self.internal_model = nn.Sequential(*internal_layers)
        self.post_model = nn.Sequential(*post_layers)
        self._weight_init()
        
    def _weight_init(self):
        # kaiming uniform init
        for layer in self.modules():
            if type(layer) is nn.Conv2d:
                weight = getattr(layer, "weight")
                nn.init.kaiming_uniform_(
                    weight,
                    a=self.NEG_SLOPE,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )

    def forward(self, x):
        pre_processed_x = self.pre_model(x)
        return (self.post_model(
                    self.internal_model(pre_processed_x) + pre_processed_x
            ))
