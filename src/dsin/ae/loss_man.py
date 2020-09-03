# https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits

import torch
from torch import nn
from torchvision.models import vgg16_bn

import numpy as np

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from dsin.ae.distortions import Distortions
from dsin.ae.si_net import SiNetChannelIn
from dsin.ae import config
from dsin.ae.kitti_normalizer import ChangeImageStatsToKitti, ChangeState


class FeatureLoss(nn.Module):
    @staticmethod
    def gram_matrix(x):
        n,c,h,w = x.size()
        x = x.view(n, c, -1)
        return (x @ x.transpose(1,2))/(c*h*w)

    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]
        self.noramlize = ChangeImageStatsToKitti(direction=ChangeState.NORMALIZE)
    
    @classmethod
    def create_loss(cls,device=None):
        if device is None and torch.cuda.is_available():
            vgg_m = vgg16_bn(True).features.cuda().eval()
            loss = cls(vgg_m,[22, 32, 42],[5,15,2])
            loss.noramlize.cuda()
        else:
            vgg_m = vgg16_bn(True).features.eval()
            loss = cls(vgg_m,[22, 32, 42],[5,15,2])
        requires_grad(vgg_m, False)

        return loss


    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        # input and target are concatenated between 0 and 1 
        # input and target should after mult by 255 have kitti stats so we remove
        # it by normalizing it.
        input = self.noramlize(input * config.open_image_normalization)
        target = self.noramlize(target * config.open_image_normalization)
        
        base_loss = F.l1_loss

        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(self.gram_matrix(f_in), self.gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


class LossManager(nn.Module):
    log_natural_base_to_base2_factor = np.log2(np.e)

    def __init__(self, model, use_side_infomation: SiNetChannelIn, target_bit_cost, use_feat_loss = False ):
        super().__init__()
        # don't average over batches, will happen after importance map mult.
        self.bit_cost_loss = nn.CrossEntropyLoss(reduction="none")
        self.si_net_loss = nn.L1Loss(reduction="mean")
        self.use_side_infomation = use_side_infomation
        
        if use_feat_loss:
            self.feat_loss = FeatureLoss.create_loss()
        self.use_feat_loss = use_feat_loss
        self.model = model
        self.importnace_map_dict=dict()
        self.target_bit_cost = target_bit_cost

       

    def forward(self, *vargs):


        if hasattr(self.model,"quantizer"): #base_ae case
            centers = self.model.quantizer.centers
        else: # si_ae case
            centers = self.model.ae.quantizer.centers

        self.centers_regularization_term = 0.5 * 0.1 * (centers ** 2).sum()


        (_,_,self.x_reconstructed,
         self.x_dec,
         x_pc,
         importance_map_mult_weights,
         x_quantizer_index_of_closest_center, self.x_orig,y,
         l2_weights,    
         ) = self.model.my_tuple


        self.importnace_map_dict['mean']= torch.mean(importance_map_mult_weights)
        self.importnace_map_dict['var']= torch.var(importance_map_mult_weights)


        # self.x_orig - orig img with color levels between 0 and 1 
       
       
        self.bit_cost_loss_value = self._get_bit_cost_loss(
            pc_output=x_pc,
            quantizer_closest_center_index=x_quantizer_index_of_closest_center,
            importance_map_mult_weights=importance_map_mult_weights,
            beta_factor=config.beta,
        )

        self.feat_loss_value = 20 * (self.feat_loss(self.x_reconstructed,  self.x_orig) if 
            self.use_feat_loss 
            and self.use_side_infomation == SiNetChannelIn.WithSideInformation
            else 0 )
        # import pdb
        # pdb.set_trace()
        self.si_net_loss_value = (
            self.si_net_loss(self.x_reconstructed,  self.x_orig) * 255.0
            if self.use_side_infomation == SiNetChannelIn.WithSideInformation
            else 0
        )


        self.autoencoder_loss_value = 255.0 * Distortions._calc_dist(
            self.x_dec,
             self.x_orig,
            distortion=config.autoencoder_loss_distortion_to_minimize,
            cast_to_int=False,
        )
        self.l2_reg_loss = l2_weights * config.l2_reg_coeff
        self.total_loss = (self.l2_reg_loss
            + self.centers_regularization_term
            + self.autoencoder_loss_value * (1 - config.si_loss_weight_alpha) 
            + self.si_net_loss_value * config.si_loss_weight_alpha
            + self.bit_cost_loss_value
            + self.feat_loss_value
        )
        return self.total_loss

    def _get_bit_cost_loss(
        self,
        pc_output,
        quantizer_closest_center_index,
        importance_map_mult_weights,
        beta_factor,
        device=None,
    ):
        """
            Parameters:
                pc_output : tensor NCHW, floats
                quantizer_closest_center_index: tensor NHW, value in [0,..,C-1]
                importance_map_mult_weights: tensor NHW
                beta_factor: float
            """
        # calculate crossentropy of the softmax of pc_output w.r.t the
        # indexes of the closest center of each pixel

        # bitcost :  (NCHW,NCHW) -> NHW
        self.cross_entropy_loss_in_bits = (
            self.bit_cost_loss(
                input=pc_output, target=quantizer_closest_center_index)
            * self.log_natural_base_to_base2_factor
        )

        mean_real_bit_entropy = torch.mean(self.cross_entropy_loss_in_bits)

        self.masked_bit_entropy = torch.mul(
            self.cross_entropy_loss_in_bits, importance_map_mult_weights
        )
        mean_masked_bit_entropy = torch.mean(self.masked_bit_entropy)

        self.soft_bit_entropy = 0.5 * \
            (mean_masked_bit_entropy + mean_real_bit_entropy)

        if device is None and torch.cuda.is_available():
            t_zero = torch.tensor(0.0, dtype=torch.float32,
                                  requires_grad=False).cuda()
        else:
            t_zero = torch.tensor(
                0.0, dtype=torch.float32, requires_grad=False)

        return beta_factor * torch.max(self.soft_bit_entropy - self.target_bit_cost, t_zero)
