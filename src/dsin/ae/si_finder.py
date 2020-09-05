import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align


class SiFinder(nn.Module):
    KERNEL_SIZE = 24
    INPUT_CHANNELS = 3
    CORR_SIGMA = 1e-6

    def create_y_syn(self, x_dec: torch.Tensor, y_dec: torch.Tensor,
                     y: torch.Tensor, device=None):
        """
        create synthetic image from patches in y.
        Let patch is places in index A inside y_syn,
        the patch is taken from y at index B.
        B is the offset of the patch in y_dec which has the maximum correlation
        to a patch in index A in x_dec.

        x_dec and y_dec should be normalized to the smame standart
        y is not required to be normalized as it patches are only assembled, but
        no calculation are peromed on them.

        x_dec: tensor, 1CH`W`
        y_dec: tensor, 1CHW
        """
        # tuple of P tuples of (row,col) patch offsets
        patch_offsets_in_y_dec = self._get_best_patch_index(
            x_dec=x_dec, y_dec=y_dec)

        patch_w = self.KERNEL_SIZE
        patch_h = self.KERNEL_SIZE

        # the 0 in the first dimension of boxes refers to the first image in the
        # batch.
        # the -0.5 offset is because roi_align refers to the center of the pixel
        # hence requirign a shift of 0.5-pixel-length.
        if device is None and torch.cuda.is_available():
            boxes = torch.tensor(
                [
                    [
                        0,
                        -0.5 + offset_w,
                        -0.5 + offset_h,
                        -0.5 + offset_w + patch_w,
                        -0.5 + offset_h + patch_h,
                    ]
                    for (offset_h, offset_w) in patch_offsets_in_y_dec
                ]
            ).cuda()

        else:
            boxes = torch.tensor(
                [
                    [
                        0,
                        -0.5 + offset_w,
                        -0.5 + offset_h,
                        -0.5 + offset_w + patch_w,
                        -0.5 + offset_h + patch_h,
                    ]
                    for (offset_h, offset_w) in patch_offsets_in_y_dec
                ]
            )

        # PCKK
        y_patches = roi_align(
            y,
            boxes,
            output_size=(patch_h, patch_w),
            spatial_scale=1.0,
            sampling_ratio=-1,
        )

        # CKKP
        y_patches = y_patches.permute(1, 2, 3, 0)

        # 1|(C*K*K)|P , nn.Fold expect on input the same shape that nn.Unfold
        # return.
        y_patches = y_patches.view(
            1, x_dec.shape[1] * self.KERNEL_SIZE *
            self.KERNEL_SIZE, y_patches.shape[-1]
        )

        sticher = self._get_patch_stiching(output_size=x_dec.shape[-2:])
        return sticher(y_patches)

    def _get_best_patch_index(self, x_dec: torch.Tensor, y_dec: torch.Tensor):
        """
        Find the index of top left corener y_dec patch with max correlation
        for each patch in x_dec.

        index returned in (row,col) fromat.

        Parameters:
            x_dec : 1CHW, x_dec to find best patches
            y_dec: 1CHW
        """
        # PCKK
        x_patches = self._get_x_patches(x_dec)

        # 1P(H-K+1)(W-K+1)
        # P - nof x patches
        # K - x patch size
        # H,W - original size of y_dec
        x_y_dec_corr = self.pearson_corr(x_patches, y_dec)
        corr_shape = x_y_dec_corr.shape

        # reshape into 1P[(H-K+1)*(W-K+1)] to allow maximization over a single dim.
        x_y_dec_corr = x_y_dec_corr.view(
            corr_shape[0], corr_shape[1], corr_shape[2] * corr_shape[3]
        )

        # drop dim=0 which is just 1
        best_patch_vector_index = torch.argmax(x_y_dec_corr[0, :, :], dim=-1)
        # divide and modulu over the width to find the row and the coloumn
        return tuple(
            (v // corr_shape[-1], v % corr_shape[-1]) for v in best_patch_vector_index
        )

    def _get_x_patches(self, x_dec: torch.Tensor):
        """Get non-overlapping patches of size self.KERNEL_SIZE of x_dec.

        patch order is raster(scan by coloums then go to the next row of
        patches).
        """
        if (
            x_dec.shape[-1] % self.KERNEL_SIZE != 0
            or x_dec.shape[-2] % self.KERNEL_SIZE != 0
        ):
            raise ValueError(
                f"x_dec.shape={x_dec.shape} is not dividable by self.KERNEL_SIZE={self.KERNEL_SIZE}"
            )
        if x_dec.shape[0] != 1:
            raise ValueError(
                f"support batch size 1 only while x_dec.shape[0]{x_dec.shape[0]}")

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

    def _get_patch_stiching(self, output_size: torch.Tensor):
        """stich non overlaping patches of size KERNEL_SIZE*KERNEL_SIZE.
        Parameters:
            output_size -tensor, dim-(1,2). represeting the |H|W| of the output
             from all stiched patches
        """
        return nn.Fold(
            output_size=output_size,
            kernel_size=self.KERNEL_SIZE,
            stride=self.KERNEL_SIZE,
        )

    @staticmethod
    def pearson_corr(x_patches: torch.tensor, y: torch.tensor, device=None):
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

        if device is None and torch.cuda.is_available():
            # kernel of dims PCKK will lead to output 1P(H-K+1)(W-K+1) with y as input
            mean_kernel = torch.ones(
                (x_patches.shape[0], y.shape[1],
                 x_patches.shape[2], x_patches.shape[3]),
                dtype=torch.float32).cuda()
        else:
            mean_kernel = torch.ones(
                (x_patches.shape[0], y.shape[1],
                 x_patches.shape[2], x_patches.shape[3]),
                dtype=torch.float32)

        # dim - 1P(H-K+1)(W-K+1)
        sum_y = F.conv2d(y, weight=mean_kernel)

        # dim - 1P(H-K+1)(W-K+1)
        # sum_of_y_square = F.conv2d(y ** 2, weight=mean_kernel)

        ##################
        # 1P(H-K+1)(W-K+1)
        # sum_xy = F.conv2d(y, weight=x_patches)

        # patch_size
        patch_size = x_patches.shape[-1] * x_patches.shape[-2] * x_patches.shape[-3] 
        # numerator
        numerator = F.conv2d(y, weight=x_patches)/patch_size- mean_x * sum_y/patch_size
        ##################
        # denominator
      
        denominator_x = sum_of_x_square/patch_size - mean_x * mean_x

        ##########
        # denominator_y = torch.sqrt(sum_of_y_square/patch_size - sum_y * sum_y / (patch_size ** 2))
        denominator_y = F.conv2d(y ** 2, weight=mean_kernel)/patch_size - sum_y * sum_y / (patch_size ** 2)
        ##########
        # denominator = denominator_x * denominator_y
        ##################
       
        # s_coeff = (numerator + SiFinder.CORR_SIGMA) / (denominator + SiFinder.CORR_SIGMA)
        # l_coeff = ( 2 * sum_y * mean_x/patch_size  + SiFinder.CORR_SIGMA)/( sum_y * sum_y/(patch_size**2) + mean_x ** 2 + SiFinder.CORR_SIGMA)
        # c_coeff = (2 * numerator + SiFinder.CORR_SIGMA)/ (denominator_x** 2 +denominator_y**2 +  + SiFinder.CORR_SIGMA)
      
        return (2 * numerator + SiFinder.CORR_SIGMA)/ (torch.abs(denominator_x) +torch.abs(denominator_y) +  SiFinder.CORR_SIGMA)
        # import pdb
        # pdb.set_trace()
        # return  (((2 * numerator + SiFinder.CORR_SIGMA)/ (denominator_x** 2 +denominator_y**2 +  + SiFinder.CORR_SIGMA)) * 
        # (( 2 * sum_y * mean_x/patch_size  + SiFinder.CORR_SIGMA)/( sum_y * sum_y/(patch_size**2) + mean_x ** 2 + SiFinder.CORR_SIGMA)))


