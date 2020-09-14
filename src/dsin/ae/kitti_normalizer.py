# originally generated from : dev_nb/nb__autoencoder_imgcomp.ipynb
import torch
from torch import nn
from enum import Enum

class ChangeState(Enum):
    NORMALIZE = 0
    DENORMALIZE = 1
    OFF = 2


class ChangeImageStatsToKitti(nn.Module):
    SIGMA_MIN = 1e-5

    def __init__(self, direction: ChangeState, input_channels: int = 3):
        super().__init__()

        self.direction = direction
        if input_channels % 3 != 0:
            raise ValueError(f"input_channels {input_channels}, should divide be 3 ")
        self.channel_repeat_factor = input_channels // 3

        mean, var = self._get_stats(self.channel_repeat_factor)

        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.direction == ChangeState.NORMALIZE:
            return self._normalize(x)
        elif self.direction == ChangeState.DENORMALIZE:
            return self._denormalize(x)
        elif self.direction == ChangeState.OFF:
            return x

        raise ValueError(f"Invalid stats change direction {self.direction}")

    def _normalize(self, x):

        return (x - self.mean) / torch.sqrt(self.var + self.SIGMA_MIN)

    def _denormalize(self, x):
        
        return self.sigmoid(x)

    @staticmethod
    def _get_stats(channel_repeat_factor: int):
        """Get mean and variance values of KITTI dataset."""
        # make mean, var into (3, 1, 1) so that they broadcast with NCHW
        mean = (
            torch.tensor(
                [93.70454143384742, 98.28243432206516, 94.84678088809876],
                dtype=torch.float32,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        var = (
            torch.tensor(
                [5411.79935676, 5758.60456747, 5890.31451232], dtype=torch.float32
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        return (
            mean.repeat(channel_repeat_factor, 1, 1),
            var.repeat(channel_repeat_factor, 1, 1),
        )



class ChangeImageStatsToImagenet(nn.Module):
    SIGMA_MIN = 1e-5

    def __init__(self, direction: ChangeState, input_channels: int = 3):
        super().__init__()

        self.direction = direction
        if input_channels % 3 != 0:
            raise ValueError(f"input_channels {input_channels}, should divide be 3 ")
        self.channel_repeat_factor = input_channels // 3

        mean, var = self._get_stats(self.channel_repeat_factor)

        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.direction == ChangeState.NORMALIZE:
            return self._normalize(x)
        elif self.direction == ChangeState.DENORMALIZE:
            return self._denormalize(x)
        elif self.direction == ChangeState.OFF:
            return x

        raise ValueError(f"Invalid stats change direction {self.direction}")

    def _normalize(self, x):

        return (x - self.mean) / torch.sqrt(self.var + self.SIGMA_MIN)

    def _denormalize(self, x):
        
        return self.sigmoid(x)

    @staticmethod
    def _get_stats(channel_repeat_factor: int):
        """Get mean and variance values of IMAGENET dataset."""
        # make mean, var into (3, 1, 1) so that they broadcast with NCHW
        mean = (
            torch.tensor(
                [123.675, 116.28 , 103.53 ],
                dtype=torch.float32,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        var = (
            torch.tensor(
                [3409.976025, 3262.6944  , 3291.890625], dtype=torch.float32
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        return (
            mean.repeat(channel_repeat_factor, 1, 1),
            var.repeat(channel_repeat_factor, 1, 1),
        )

