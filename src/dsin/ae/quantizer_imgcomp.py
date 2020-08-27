import torch
import torch.nn.functional as F
from torch import nn
from enum import Enum, auto


"""Quantizer layer."""


class ChannelOrder(Enum):
    NHWC = auto()
    NCHW = auto()


class Quantizer(nn.Module):
    def __init__(
        self,
        num_centers: int,
        centers_initial_range,
        centers_regularization_factor: float,
        sigma: float,
        init_centers_uniformly: bool = False,
    ):
        super().__init__()
        self.num_centers = (num_centers,)
        self.centers_initial_range = centers_initial_range
        self.sigma = sigma
        self.reg = torch.as_tensor(centers_regularization_factor, dtype=torch.float32)
        self.init_centers_uniformly = init_centers_uniformly
        self._create_centers_variable()

    def _create_centers_variable(self, dtype=torch.float32):  # (C, L) or (L,)

        minval, maxval = map(int, self.centers_initial_range)
        # create a tensor of size with values drawn
        # from uniform distribution
        if self.init_centers_uniformly:
            centers = torch.linspace(
                start=self.centers_initial_range[0],
                end=self.centers_initial_range[1],
                steps=self.num_centers[0],
                dtype=dtype,
            )
        else:
            centers = (
                torch.rand(*self.num_centers, dtype=dtype) * (maxval - minval) + minval
            )
            centers, _ = torch.sort(centers)
        # Wrapping with nn.Parameter ensures it is copied to gpu when .to('cuda') is called
        self.centers = nn.Parameter(centers)

    def get_centers_regularization_term(self):
        # calculate half the l2 norm  like tf.nn.l2_loss(centers)
        # return 0.5 * self.reg * (self.centers ** 2).sum().cpu()
        return 0.5 * self.reg * torch.nn.MSELoss()(self.centers,self.centers.new_zeros(self.centers.shape))

    def __repr__(self):
        return f"{self.__class__.__name__}(sigma={self.sigma})"

    def forward(self, x, dataformat: ChannelOrder = ChannelOrder.NCHW):
        assert x.dtype == torch.float32, "x should be float32"
        assert self.centers.dtype == torch.float32, "centers should be float32"
        assert (
            len(x.size()) == 4
        ), f"x should be NCHW or NHWC got {len(x.size())} instead"
        assert (
            len(self.centers.shape) == 1
        ), f"centers should be (L,), got {len(a.centers.size())}"

        # improve numerics by calculating using NCHW
        if dataformat == ChannelOrder.NHWC:
            x = self.permute_NHWC_to_NCHW(x)

        x_soft, x_hard, x_index_of_closest_center = self._quantize(x)

        # return tensors in the original channel order
        if dataformat == ChannelOrder.NHWC:
            return tuple(map(self.permute_NCHW_to_NHWC, (x_soft, x_hard, x_index_of_closest_center)))
        else:
            return x_soft, x_hard, x_index_of_closest_center

    def _quantize(self, x):

        N, C, H, W = x.shape

        # Turn each image into vector, i.e. make x into NCm1, where m=H*W
        x = x.view(N, C, H * W, 1)
        # shape- NCmL, calc distance to l-th center
        d = torch.pow(x - self.centers, 2)
        # shape- NCmL, \sum_l d[..., l] sums to 1
        phi_soft = F.softmax(-self.sigma * d, dim=-1)
        # - Calcualte soft assignements ---
        # NCm, soft assign x to centers
        x_soft = torch.sum(self.centers * phi_soft, dim=-1)
        # NCHW
        x_soft = x_soft.view(N, C, H, W)

        ######################
        # Calcualte hard assignements for the forward pass, keep gards for the backward
        ######################

        # NCm, symbols_hard[..., i] contains index of symbol closest to each pixel
        # detach d to use values without affecting the gradients
        _, symbols_hard = torch.min(d.detach(), dim=-1)
        # NCHW
        x_index_of_closest_center = symbols_hard.view(N, C, H, W)
        # NCHW, contains value of symbol to use
        x_hard = self.centers[x_index_of_closest_center]

        x_soft.data = x_hard  # assign data, keep gradient
        return x_soft, x_hard, x_index_of_closest_center

    @staticmethod
    def permute_NHWC_to_NCHW(t):
        N, H, W, C = 0, 1, 2, 3
        return t.permute(N, C, H, W)

    @staticmethod
    def permute_NCHW_to_NHWC(t):
        N, C, H, W = 0, 1, 2, 3
        return t.permute(N, H, W, C)
