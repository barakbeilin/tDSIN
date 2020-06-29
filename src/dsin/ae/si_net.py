import torch
from torch import nn
from dsin.ae import config


class SiNet(nn.Module):
    def __init__(self):
        super().__init__()
        NOF_INTERNAL_LAYERS = 7
        NEG_SLOPE = 0.2

        internal_layers = [
            (
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=[3, 3],
                    dilation=2 ** (i + 1),
                    padding_mode="replicate",
                    padding=[2 ** (i + 1), 2 ** (i + 1)],
                ),
                nn.LeakyReLU(negative_slope=NEG_SLOPE),
            )
            for i in range(NOF_INTERNAL_LAYERS)
        ]

        internal_layers = sum(internal_layers, ())
        layers = [
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=[3, 3],
                padding_mode="replicate",
                padding=[1, 1],
            ),
            nn.LeakyReLU(negative_slope=NEG_SLOPE),
            *internal_layers,
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=[1, 1],),
        ]

        for layer in layers:
            try:
                weight = getattr(layer, "weight")
                # swapped former to kaiming uniform
                nn.init.kaiming_uniform_(
                    weight, a=NEG_SLOPE, mode="fan_in", nonlinearity="leaky_relu"
                )
            except AttributeError:
                pass

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

