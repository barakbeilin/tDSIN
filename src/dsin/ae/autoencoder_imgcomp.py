# originally generated from : dev_nb/nb__autoencoder_imgcomp.ipynb
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List
import enum
from enum import Enum

class ChangeState(Enum):
    NORMALIZE=enum.auto(),
    DENORMALIZE=enum.auto(),
    OFF= enum.auto()

class ChangeImageStatsToKitti(nn.Module):
    SIGMA_MIN = 1e-5
    def __init__(self, direction: ChangeState):
        super().__init__()
        self.direction = direction

        mean, var = self._get_stats()
        self.mean, self.var = nn.Parameter(mean), nn.Parameter(var)

    def forward(self, x):
        if self.direction == ChangeState.NORMALIZE:
            return self._normalize(x)
        elif self.direction == ChangeState.DENORMALIZE:
            return self._denormalize(x)
        elif self.direction == ChangeState.OFF:
            return x
        else:
            raise ValueError(f'Invalid stats change direction {self.direction}')

    def _normalize(self, x):

        return (x - self.mean) / torch.sqrt( self.var + self.SIGMA_MIN)

    def _denormalize(self, x):

        return (x * torch.sqrt(self.var +  self.SIGMA_MIN5)) + self.mean

    @staticmethod
    def _get_stats():
        """Get mean and variance values of KITTI dataset."""
        # make mean, var into (3, 1, 1) so that they broadcast with NCHW
        mean = torch.nn.Parameter(
            torch.tensor([93.70454143384742, 98.28243432206516, 94.84678088809876],
                         dtype=torch.float32), requires_grad=False).unsqueeze(-1).unsqueeze(-1)
        var = torch.nn.Parameter(
            torch.tensor([5411.79935676, 5758.60456747, 5890.31451232],
                         dtype=torch.float32), requires_grad=False).unsqueeze(-1).unsqueeze(-1)
        return mean, var

class Enc_Cs:
    """Consts for encoder"""
    in_channels_to_conv2d_1= 3
    n = 128
    quantizer_num_of_centers = 32
    use_heat_map = True

    basic_conv2d={'padding_mode':'replicate'}
    modifiable_conv2d={'conv':nn.Conv2d, **basic_conv2d}
    padding_stride2_kernel5={'kernel_size': [5,5],'stride': [2,2],'padding':[2,2]}
    padding_stride1_kernel3={'kernel_size': [3,3],'stride': [1,1],'padding':[1,1]}

    enc_conv2d_1={'in_channels': in_channels_to_conv2d_1,
                  'out_channels': n // 2,
                  **padding_stride2_kernel5,
                  **modifiable_conv2d}
    enc_conv2d_2={'in_channels': n // 2, 'out_channels': n,
                  **padding_stride2_kernel5,
                  **modifiable_conv2d}

    enc_resblock={'in_channels': n , 'out_channels': n,
                  **padding_stride1_kernel3,
                  **modifiable_conv2d}

    enc_uber_resblock={'num_of_resblocks':3, 'resblock':enc_resblock}
    enc_uber_resblocks={'num_of_uberresblocks': 5, 'uberresblock':enc_uber_resblock}

    last_conv2d_out =  \
        quantizer_num_of_centers + 1 if use_heat_map else quantizer_num_of_centers
    last_conv2d={'in_channels': n,
                 'out_channels': last_conv2d_out,
                  **padding_stride2_kernel5,
                  **basic_conv2d}

class Dec_Cs(Enc_Cs):
    def __init__(self):
        super().__init__()
        #override ```sself.basic_conv2d``` since nn.ConvTranspose2d cam padd only with zeros
        self.basic_conv2d={'padding_mode':'zeros'}
        #override ```self.modifiable_conv2d``` from nn.Conv2d to nn.ConvTranspose2d
        self.modifiable_conv2d={'conv':nn.ConvTranspose2d, **self.basic_conv2d}
        self.padding_stride2_kernel3={'kernel_size': [3,3],'stride': [2,2],'padding':[1,1]}
        self.conv2d_1 = {'in_channels': self.quantizer_num_of_centers,
                      'out_channels': self.n ,
                      **self.padding_stride2_kernel3,
                      **self.modifiable_conv2d}


        self.dec_resblock = {'in_channels': self.n , 'out_channels': self.n,
                  **self.padding_stride1_kernel3,
                  **super().modifiable_conv2d}
        self.dec_uber_resblock = {'num_of_resblocks': 3, 'resblock':self.dec_resblock}
        self.dec_uber_resblocks = {'num_of_uberresblocks': 5, 'uberresblock':self.dec_uber_resblock}

        self.dec_prelast_conv2d = {'in_channels': self.n , 'out_channels': self.n // 2,
                  **self.padding_stride2_kernel5,
                  **self.modifiable_conv2d}

        self.dec_last_conv2d = {'in_channels': self.n // 2 , 'out_channels': 3,
                  **self.padding_stride2_kernel5,
                  **self.basic_conv2d}

class Encoder(nn.Module):
    # test encoder
    # enc = Encoder.create_module_from_const()
    # txt = 'Encoder(\n  (pre_sum_model): Sequential(\n    (0): ChangeImageStatsToKitti()\n    (1): Conv2dReluBatch2d(\n      (model): Sequential(\n        (0): Conv2d(3, 64, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], padding_mode=replicate)\n        (1): ReLU()\n        (2): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n      )\n    )\n    (2): Conv2dReluBatch2d(\n      (model): Sequential(\n        (0): Conv2d(64, 128, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], padding_mode=replicate)\n        (1): ReLU()\n        (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n      )\n    )\n    (3): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (4): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (5): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (6): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (7): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (8): ResBlock(\n      (model): Sequential(\n        (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n        (1): ReLU()\n        (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n        (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n      )\n    )\n  )\n  (post_sum_model): Conv2d(128, 33, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], padding_mode=replicate)\n)'
    # assert txt==repr(enc),"Encoder doesn't comply article"
    # print(enc)

    def __init__(self,conv2d_1: Dict,
                      conv2d_2: Dict,
                      uberresblocks: Dict,
                      prelast_resblock: Dict,
                      last_conv2d: Dict):
        super().__init__()
        layers = [ChangeImageStatsToKitti(direction=ChangeState.NORMALIZE)]

        # first conv layers
        layers.extend([Conv2dReluBatch2d(**conv2d_1),
                       Conv2dReluBatch2d(**conv2d_2)])

        # 5 uber blocks
        for i in range(uberresblocks['num_of_uberresblocks']):
            layers.append(UberResBlock(**uberresblocks['uberresblock']))

        # resblock after the last uber-block
        layers.append(ResBlock(**prelast_resblock))
        self.pre_sum_model = nn.Sequential(*layers)

        self.post_sum_model= nn.Conv2d(**last_conv2d)

    @classmethod
    def create_module_from_const(cls):
        return cls(Enc_Cs.enc_conv2d_1,
                       Enc_Cs.enc_conv2d_2,
                       Enc_Cs.enc_uber_resblocks,
                       Enc_Cs.enc_uber_resblocks['uberresblock']['resblock'],
                       Enc_Cs.last_conv2d)




    def forward(self, x):
        res_pre_conv = self.pre_sum_model(x) + x
        return self.post_sum_model(res_pre_conv)

class UberResBlock(nn.Module):

    def __init__(self,num_of_resblocks: int, resblock: Dict):
        super().__init__()
        layers = []
        for i in range(num_of_resblocks):
            layers.append(ResBlock(**resblock))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) + x

class ResBlock(nn.Module):
    NUM_OF_CONVS = 2
    def __init__(self,in_channels: int, out_channels: int, kernel_size: List,
                 stride: List,padding_mode: str,padding, conv, num_of_convs: int = NUM_OF_CONVS):
        super().__init__()
        layers = []
        for i in range(num_of_convs):
            layers.append(conv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding_mode=padding_mode,
                                    padding=padding))
            if i==0:
                layers.extend([nn.ReLU(),
                nn.BatchNorm2d(out_channels, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) + x

class Conv2dReluBatch2d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: List,
                 stride: List,padding_mode: str,padding, conv):
        super().__init__()


        self.model = nn.Sequential(conv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding_mode=padding_mode,
                                    padding=padding),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_channels,
                                                  eps=1e-03,
                                                  momentum=0.1,
                                                  affine=True,
                                                  track_running_stats=True))



    def forward(self, x):
        return self.model(x)



class Decoder(nn.Module):
    # >>> test_decoder
    # dec = Decoder.create_module_from_const()
    # txt= 'Decoder(\n  (pre_uberblock_model): Conv2dReluBatch2d(\n    (model): Sequential(\n      (0): ConvTranspose2d(32, 128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])\n      (1): ReLU()\n      (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n    )\n  )\n  (pre_sum_model): Sequential(\n    (0): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (1): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (2): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (3): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (4): UberResBlock(\n      (model): Sequential(\n        (0): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (1): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n        (2): ResBlock(\n          (model): Sequential(\n            (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n            (1): ReLU()\n            (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n            (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n          )\n        )\n      )\n    )\n    (5): ResBlock(\n      (model): Sequential(\n        (0): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n        (1): ReLU()\n        (2): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n        (3): Conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], padding_mode=replicate)\n      )\n    )\n  )\n  (post_sum_model): Sequential(\n    (0): Conv2dReluBatch2d(\n      (model): Sequential(\n        (0): ConvTranspose2d(128, 64, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2])\n        (1): ReLU()\n        (2): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)\n      )\n    )\n    (1): ConvTranspose2d(64, 3, kernel_size=[5, 5], stride=[2, 2], padding=[2, 2])\n    (2): ChangeImageStatsToKitti()\n  )\n)'
    # assert repr(dec) == txt, "Decoder doesn't comply article"
    # print(dec)

    def __init__(self,conv2d_1: Dict,
                      uberresblocks: Dict,
                      post_uberblock_resblock: Dict,
                      prelast_conv2d: Dict,
                      last_conv2d: Dict):
        super().__init__()
        layers = []

        # first deconv layers
        self.pre_uberblock_model = Conv2dReluBatch2d(**conv2d_1)

        # 5 uber blocks
        for i in range(uberresblocks['num_of_uberresblocks']):
            layers.append(UberResBlock(**uberresblocks['uberresblock']))
        # resblock after the last uber-block
        layers.append(ResBlock(**post_uberblock_resblock))
        self.pre_sum_model = nn.Sequential(*layers)

        self.post_sum_model= nn.Sequential(Conv2dReluBatch2d(**prelast_conv2d),
                                           nn.ConvTranspose2d(**last_conv2d),
                                           ChangeImageStatsToKitti(direction=ChangeState.DENORMALIZE))

    @classmethod
    def create_module_from_const(cls):
        consts = Dec_Cs()
        return cls(consts.conv2d_1,
                       consts.dec_uber_resblocks,
                       consts.dec_resblock,
                       consts.dec_prelast_conv2d,
                       consts.dec_last_conv2d)


    def forward(self, x):
        pre_ubberblock = self.pre_uberblock_model(x)
        res_pre_conv = self.pre_sum_model(pre_ubberblock) + pre_ubberblock
        return self.post_sum_model(res_pre_conv)