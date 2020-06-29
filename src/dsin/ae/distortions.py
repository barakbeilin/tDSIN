import torch
from torch import nn
from enum import Enum
from pytorch_msssim import ms_ssim
from dsin.ae import config

"""https://discuss.pytorch.org/t/use-model-parameters-in-loss-function/17305"""
class DistTypes(Enum):
    MAE = 0
    MSE = 1
    PSNR = 2
    MS_SSMIM = 3


class PSNRLoss:
    def __init__(self):
        self.mse = nn.MSELoss()

    def __call__(self, x: torch.tensor, target: torch.tensor) -> torch.tensor:
        return 10 * torch.log10(255.0 * 255.0 / self.mse(x, target))


class Distortions:
    def __init__(
        self,
        x: torch.tensor,
        x_out: torch.tensor,
        distortion_to_minimize: DistTypes,
        is_training: bool,
    ):
        self.K_MS_SSIM = 5000
        self.K_PSNR = 100
        self.distortion_to_minimize = distortion_to_minimize

        # If not training, always cast to int, because we don't need the
        # gradients.
        # if we don't minimize for PSNR(MSE), cast x and x_out to int before
        # calculating the PSNR(MSE), because otherwise PSNR is off
        cast_to_int_for_psnr = (
            not is_training
        ) or self.distortion_to_minimize != DistTypes.PSNR

        cast_to_int_for_mse = (
            not is_training
        ) or self.distortion_to_minimize != DistTypes.MSE

        cast_to_int_for_mae = (
            not is_training
        ) or self.distortion_to_minimize != DistTypes.MAE

        self.mae = self._calc_dist(x, x_out, DistTypes.MAE, cast_to_int_for_mae)
        self.mse = self._calc_dist(x, x_out, DistTypes.MSE, cast_to_int_for_mse)
        self.psnr = self._calc_dist(x, x_out, DistTypes.PSNR, cast_to_int_for_psnr)

        # don't calculate MS-SSIM if not necessary to speed things up
        self.ms_ssim = (
            self._calc_dist(x, x_out, DistTypes.MS_SSMIM, False)
            if self.distortion_to_minimize == DistTypes.MS_SSMIM
            else None
        )

    def get_distortion(self, distortion_type: DistTypes):
        """ Returns a float32 that should be minimized in training. For PSNR and MS-SSIM, which increase for a
        decrease in distortion, a suitable factor is added. """
        if distortion_type == DistTypes.MAE:
            return self.mae
        if distortion_type == DistTypes.MSE:
            return self.mse
        if distortion_type == DistTypes.PSNR:
            return self.K_PSNR - self.psnr
        if distortion_type == DistTypes.MS_SSMIM:
            if self.distortion_to_minimize != DistTypes.MS_SSMIM:
                raise ValueError("MSSIM not calculated if not used for minimization")
            return self.K_MS_SSIM * (1 - self.ms_ssim)

        raise ValueError("Invalid: {}".format(distortion_type))

    @staticmethod
    def _calc_dist(
        x: torch.tensor, target: torch.tensor, distortion: DistTypes, cast_to_int: bool,
    ) -> torch.tensor:
        if cast_to_int:
            # cast to int then to float since losses don't work with int.
            x_l = x.type(torch.IntTensor).type(torch.FloatTensor)
            target_l = target.type(torch.IntTensor).type(torch.FloatTensor)
        else:
            x_l, target_l = x, target

        if distortion == DistTypes.MAE:
            loss = nn.L1Loss()
        if distortion == DistTypes.MSE:
            loss = nn.MSELoss()
        if distortion == DistTypes.PSNR:
            loss = PSNRLoss()
        if distortion == DistTypes.MS_SSMIM:
            loss = ms_ssim

        return loss(x_l, target_l)


def get_loss(auto_encoder, probability_classiffier, d_loss_scaled, bit_count, heatmap):
    """
    THIS IS A UTILITY WHICH SHOULD BE MOVED ANOTHER PLACE + ADD TESTS TO IT.
    """
    heatmap_enabled = heatmap is not None
    bit_count_mask = (bit_count * heatmap) if heatmap_enabled else bit_count

    # estimated entropy calculation
    H_real = torch.mean(bit_count)
    H_mask = torch.mean(bit_count_mask)
    H_soft = 0.5 * (H_mask + H_real)

    H_target = config.H_target
    beta = config.beta

    # probability-classifier (pc)
    pc_loss = beta * torch.max(
        H_soft - H_target, torch.tensor(0.0, dtype=torch.float32)
    )

    # # Adding Regularizers
    # with tf.name_scope("regularization_losses"):
    #     reg_probclass = pc.regularization_loss()
    #     if reg_probclass is None:
    #         reg_probclass = 0
    #     reg_enc = ae.encoder_regularization_loss()
    #     reg_dec = ae.decoder_regularization_loss()
    #     reg_loss = reg_probclass + reg_enc + reg_dec

    pc_comps = [
        ("H_mask", H_mask),
        ("H_real", H_real),
        ("pc_loss", pc_loss),
    ]
    ae_comps = [("d_loss_scaled", d_loss_scaled)]

    total_loss = d_loss_scaled + pc_loss  #+ reg_loss
    return total_loss, H_real, pc_comps, ae_comps
