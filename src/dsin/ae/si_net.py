import torch
from torch import nn
from enum import Enum
from dsin.ae.kitti_normalizer import ChangeImageStatsToKitti, ChangeState


class SiNetChannelIn(Enum):
    WithSideInformation = 6
    NoSideInformation = 3

class DilatedResBlock(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,dilation,negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.layers = [nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=[3, 3],
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

    def forward(self,x):
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

                
class SiNet(nn.Module):
    NOF_INTERNAL_LAYERS = 7
    NEG_SLOPE = 0.2

    def __init__(self, in_channels: SiNetChannelIn, use_eye_init: bool = False):
        super().__init__()

        internal_layers = [
            DilatedResBlock(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=[3, 3],
                    dilation=2 ** (i + 1),
                    negative_slope =self.NEG_SLOPE )
            for i in range(self.NOF_INTERNAL_LAYERS)]

        # internal_layers = sum(internal_layers, ())
        self.layers = [
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
            *internal_layers,
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=[1, 1],),
            ChangeImageStatsToKitti(direction=ChangeState.DENORMALIZE),
        ]
        self._weight_init(use_eye_init)
        self.model = nn.Sequential(*self.layers)

    def _weight_init(self, use_eye_init):
        if use_eye_init:
            for layer in self.layers:
                if type(layer) is nn.Conv2d:
                    with torch.no_grad():
                        w = torch.eye(
                            n=layer.kernel_size[0], dtype=torch.float32
                        ).repeat(layer.weight.shape[0], layer.weight.shape[1], 1, 1)

                        layer.weight = nn.Parameter(w)

        else:
            # kaiming uniform init
            for layer in self.layers:
                if type(layer) is nn.Conv2d:
                    weight = getattr(layer, "weight")
                    nn.init.kaiming_uniform_(
                        weight,
                        a=self.NEG_SLOPE,
                        mode="fan_in",
                        nonlinearity="leaky_relu",
                    )
               
    def forward(self, x):
        return self.model(x)