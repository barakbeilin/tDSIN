import torch
from torch import nn
from torch.nn import functional as F


class SiFinder(nn.Module):
    KERNEL_SIZE = 24
    INPUT_CHANNELS = 3

    def create_y_syn(self, x_dec: torch.Tensor, y_dec: torch.Tensor, y: torch.Tensor):
        pass

    def _find_best_patch_index(self, x_dec: torch.Tensor, y_dec: torch.Tensor):
        """
        Fsind the patch index in (row,col) fromat of y_dec patch with max
        correlation for each patch in x_dec

        Parameters:
            x_dec : 1CHW, x_dec to find best patches
            y_dec: 1CHW
        """
        # PCKK
        x_patches = self._get_x_patches()

        # 1P(H-K)(W-K)
        # P - nof x patches
        # K - x patch size
        # H,W - original size of y_dec
        x_y_dec_corr = self.pearson_corr(x_patches, y_dec)
        corr_shape = x_y_dec_corr.shape

        # reshape into 1P[(H-K)*(W-K)] to allow maximization over a single dim.
        x_y_dec_corr = x_y_dec_corr.view(
            corr_shape[0], corr_shape[1], corr_shape[2] * corr_shape[3]
        )
        best_patch_vector_index = torch.argmax(x_y_dec_corr, dim=-1)

        return tuple(
            (v // x_patches.shape[-1], v % x_patches.shape[-1])
            for v in best_patch_vector_index
        )

    def _get_x_patches(self, x_dec: torch.Tensor):
        """ Get nonoverlapping patches of size self.KERNEL_SIZE of x_dec."""
        if (
            x_dec.shape[-1] % self.KERNEL_SIZE != 0
            or x_dec.shape[-2] % self.KERNEL_SIZE != 0
        ):
            raise ValueError(f"{x_dec.shape=} is not divided by {self.KERNEL_SIZE=}")
        if x_dec.shape[0] != 1:
            raise ValueError(f"support batch size 1 only while {x_dec.shape[0]=}")

        patch_creator = self._get_patch_creator()

        # 1CKKP ,
        # C - input channels of x
        # K - kernel size
        # P - number of patches
        x_patches = patch_creator(x_dec)
        x_patches = x_patches.view(
            1,
            self.INPUT_CHANNELS,
            self.KERNEL_SIZE,
            self.KERNEL_SIZE,
            x_patches.shape[-1],
        )
        # remove unncessary dim0 of size 1, move P dimension to be first
        # patches dim is PCKK to match weights of conv2d

        # conv2d(y) starts with 3 channels rgb(lab) and after x_patches weights
        # ends with P channels, the result of convolving y with each of patches
        # in x_patches
        x_patches = x_patches[0, :, :, :, :].permute(3, 0, 1, 2)
        # return dim PCKK (out-channels | in-channels| kernel-h | kernel-w)
        return x_patches

    def _get_patch_creator(self):
        # non overlaping patches of size KERNEL_SIZE*KERNEL_SIZE
        return nn.Unfold(kernel_size=self.KERNEL_SIZE, stride=self.KERNEL_SIZE)

    @staticmethod
    def pearson_corr(x_patches: torch.tensor, y: torch.tensor):
        """
        Calculate pearson_corr between patches of x_patches(PCKK) and y(1CHW).

        R =  numerator/ denominator.
        where:
        numerator = sum_i(xi*yi - y_mean*xi - x_mean*yi + x_mean*y_mean)
                  = sum_i(xi*yi) - (K^2)x_mean*y_mean
        denominator = sqrt( sum_i(xi^2 - 2xi*x_mean + x_mean^2)*sum_i(yi^2 - 2yi*y_mean + y_mean^2) )
                    = sqrt( sum_i(xi^2) - (K^2)*x_mean^2)*sqrt( sum_i(yi^2) - (K^2)*y_mean^2)
        """
        ################## mean_x, sum_of_x_square
        # mean over height and width, leaving result of dim -1P11
        mean_x = (
            torch.mean(x_patches, dim=(1, 2, 3))
            .unsqueeze_(0)
            .unsqueeze_(-1)
            .unsqueeze_(-1)
        )

        x_square = x_patches ** 2
        # dim - 1P11
        sum_of_x_square = (
            torch.sum(x_square, dim=(1, 2, 3))
            .unsqueeze_(0)
            .unsqueeze_(-1)
            .unsqueeze_(-1)
        )

        ##################
        ################## sum_y, sum_of_y_square
        y_square = y ** 2

        # kernel of dims PCKK will lead to output 1P(H-K+1)(W-K+1) with y as input
        mean_kernel = torch.ones(
            (x_patches.shape[0], y.shape[1], x_patches.shape[2], x_patches.shape[3]),
            dtype=torch.float32,
        )

        # dim - 1P(H-K+1)(W-K+1)
        sum_y = F.conv2d(y, weight=mean_kernel)

        # dim - 1P(H-K+1)(W-K+1)
        sum_of_y_square = F.conv2d(y_square, weight=mean_kernel)

        ##################
        ################## sum_xy
        # 1P(H-K+1)(W-K+1)
        sum_xy = F.conv2d(y, weight=x_patches)

        ################## patch_size
        patch_size = x_patches.shape[-1] * x_patches.shape[-2]
        ################## numerator
        numerator = sum_xy - mean_x * sum_y
        ##################
        ################## denominator
        denominator_x = torch.sqrt(sum_of_x_square - patch_size * mean_x * mean_x)
        denominator_y = torch.sqrt(sum_of_y_square - sum_y * sum_y / patch_size)
        denominator = denominator_x * denominator_y
        ##################

        return numerator / denominator

