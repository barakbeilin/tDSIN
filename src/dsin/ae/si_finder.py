import torch
from torch import nn


class SiFinder(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor, new_y: bool):
        """
        Parameters:
            x : x_dec to find best patches for
            y : multiple x_dec will be tested with same y so no need to load each
            foward pass a new buffer with same y thefore use the boolean
            new_y : load new y or use cached
        """
        return x
