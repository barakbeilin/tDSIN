import torch
from torch import nn
from torch.nn import functional as F


class SiFinder(nn.Module):
    KERNEL_SIZE = 24
    INPUT_CHANNELS = 3

    def create_y_syn(self, x_dec: torch.Tensor, y_dec: torch.Tensor, y: torch.Tensor):
        pass

    def _find_best_patch(
        self, x_dec: torch.Tensor, y_dec: torch.Tensor, patch_creator: nn.Unfold
    ):
        """
        Parameters:
            x_dec : 1CHW, x_dec to find best patches
            y_dec: 1CHW
        """

        # iter through x_dec pathces to find the best match in y_dec
        y_dec_path_indexs = []
        for i in range(x_patches.size()[-1]):
            corr_res = F.conv2d(y_dec, weight=x_patches, stride=24)
            best_index = torch.argmax(corr_res)
            # handle multiple max values
            if len(best_index.shape):
                y_dec_path_indexs.append(best_index.item())
            else:
                y_dec_path_indexs.append(best_index[0].item())

        return y_dec_path_indexs

    def _get_x_patches(self, x_dec: torch.Tensor, y_dec: torch.Tensor):
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
        return x_patches

    def _get_patch_creator(self):
        # non overlaping patches of size KERNEL_SIZE*KERNEL_SIZE
        return nn.Unfold(kernel_size=self.KERNEL_SIZE, stride=self.KERNEL_SIZE,)

    def pearson_corr(x_patches: torch.tensor, y_dec: torch.tensor):
        """
        Calculate pearson_corr between patches of x_patches(PCKK) and y(1CHW).

        R =  numerator/ denominator.
        where:
        numerator = sum_i(xi*yi - y_mean*xi - x_mean*yi + x_mean*y_mean)
                  = sum_i(xi*yi) - (2KK-1)x_mean*y_mean
        denominator = sqrt( sum_i(xi^2 - 2xi*x_mean + x_mean^2)*sum_i(yi^2 - 2yi*y_mean + y_mean^2) )
                    = sqrt( sum_i(xi^2) - (2KK-1)x_mean^2)*sqrt( sum_i(yi^2) - (2KK-1)y_mean^2)) )
        """
        ################## x_mean, y_mean

        ##################
        ################## sum_i(xi*yi)
        # 1P(H-K)(W-K)
        xy = F.conv2d(y_dec, weight=x_patches)
        ##################
        ################## sum_i(xi^2) , sum_i(yi^2)
        x_square = x_patches ** 2
        y_square = y ** 2

        # mean over height and width, leaving result of dim -P
        mean_x_square = torch.mean(x_patches, dim=(1, 2, 3))

        # kernel of dims PCKK will lead to output 1P(H-K)(W-K) with y as input
        mean_kernel = torch.ones(
            (x_patches.shape[0], y.shape[1], x_patches.shape[2], x_patches.shape[3]),
            dtype=torch.float32,
        )
        mean_y_square = F.conv2d(y_square, weight = mean_kernel))
        ##################
        ################## x_mean*sum_i(xi) , y_mean*sum_i(yi)
        x_mean_x = torch.nul(x_mean, sum_x)
        y_mean_y = tf.multiply(y_mean, sum_y)

        # x_mean^2 , y_mean^2
        x_mean_square = tf.square(x_mean)
        y_mean_square = tf.square(y_mean)




 
