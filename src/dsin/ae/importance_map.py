# originally generated from : dev_nb/nb__importance_map.ipynb
import torch
from torch import nn


class MinMaxMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that IS used
        to stash information for backward computation.
        """
        ctx.save_for_backward(x)
        return torch.max(
            torch.min(x, torch.tensor(1.0, dtype=torch.float32, requires_grad=False)),
            torch.tensor(0.0, dtype=torch.float32, requires_grad=False),
        )  # NCHW

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input  # identity grad


class ImportanceMapMult(nn.Module):
    # test module
    #    x =torch.round(torch.rand([10,10,5,5])*10)
    #    model = ImportanceMapMult()
    #    print(x.shape)
    #    y= model(x)
    #    # print(x)
    #    print(y.shape)

    def __init__(self, use_map=True):
        super().__init__()
        self.use_map = use_map

    def forward(self, x):
        """
        forward prop.
        Parameters:
            x : tensor including z-hat with importance map channel.
        """
        if not self.use_map:
            return x

        MAP_CHANNEL = 0  # if changed need to fix indexing further in the class

        # assume NCHW so channel dim number is 1
        CHANNEL_DIM = 1
        INFO_CHANNELS = x.shape[CHANNEL_DIM] - 1  # substract importance map

        cl_param = nn.Parameter(
            torch.arange(
                start=0, end=INFO_CHANNELS, dtype=torch.float32, requires_grad=False
            )
        )

        cl_param = torch.reshape(cl_param, (INFO_CHANNELS, 1, 1))

        # choose the first channel as the importance map
        importance_map = x[:, MAP_CHANNEL, ...]  # NHW
        importance_map = torch.sigmoid(importance_map) * INFO_CHANNELS
        
        importance_map.unsqueeze_(CHANNEL_DIM)  # N1HW

        z = x[:, MAP_CHANNEL + 1 :, ...]

        diff = importance_map - cl_param

        out_map = MinMaxMap.apply(diff)

        return torch.mul(out_map, z)
