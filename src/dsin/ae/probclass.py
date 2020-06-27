# originally generated from: dev_nb/nb__probclass.ipynb

import torch
import torch.nn.functional as F
from torch import nn
from enum import Enum, auto


#  def bitcost(self, q, target_symbols, is_training, pad_value=0):
#         """
#         Pads q, creates PC network, calculates cross entropy between output of PC network and target_symbols
#         :param q: NCHW
#         :param target_symbols:
#         :param is_training:
#         :return: bitcost per symbol: NCHW
#         """
#         tf_helpers.assert_ndims(q, 4)

#         with self._building_ctx(self.reuse):
#             if self.first_mask is None:
#                 self.first_mask = self.create_first_mask()  # DHWio
#                 self.other_mask = self.create_other_mask()  # DHWio

#             self.reuse = True

#             targets_one_hot = tf.one_hot(target_symbols, depth=self.L, axis=-1, name='target_symbols')

#             q_pad = pad_for_probclass3d(
#                     q, context_size=self.get_context_size(self.config),
#                     pad_value=pad_value, learn_pad_var=False)
#             with tf.variable_scope('logits'):
#                 # make it into NCHWT, where T is the channel dim of the conv3d
#                 q_pad = tf.expand_dims(q_pad, -1, name='NCHWT')
#                 logits = self._logits(q_pad, is_training)

#             if self.config.regularization_factor is not None:
#                 print('Creating PC regularization...')
#                 weights = _get_all_conv3d_weights_in_scope(self._PROBCLASS_SCOPE)
#                 assert len(weights) > 0
#                 reg = self.config.regularization_factor * tf.add_n(list(map(tf.nn.l2_loss, weights)))
#                 tf.losses.add_loss(reg, tf.GraphKeys.REGULARIZATION_LOSSES)

#             if targets_one_hot.shape.is_fully_defined() and logits.shape.is_fully_defined():
#                 tf_helpers.assert_equal_shape(targets_one_hot, logits)

#             with tf.name_scope('bitcost'):
#                 # softmax_cross_entropy_with_logits is basis e, change base to 2
#                 log_base_change_factor = tf.constant(np.log2(np.e), dtype=tf.float32)
#                 bc = tf.nn.softmax_cross_entropy_with_logits(
#                     logits=logits, labels=targets_one_hot) * log_base_change_factor  # NCHW

#             return bc # bit cost


class MaskedConv3d(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size, filter_mask):
        """
        custom config of conv3d:
        - use VALID padding (i.e no padding)
        - weight initz - xaviel_init
        - bias init - zero_init
        """
        super().__init__()
        self.filter_mask = nn.Parameter(filter_mask)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels ,
                        kernel_size=kernel_size, bias= True)

        # initalize layer
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)



    def forward(self,x):
        self._mask_conv_filter()
        return self.conv(x)

    def _mask_conv_filter(self):
        with torch.no_grad():
            print((self.conv.weight.shape))
            print((self.filter_mask.shape))
            self.conv.weight = self.conv.weight * self.filter_mask

    @staticmethod
    def create_maskA(kernel_size, filter_shape):
            """create 5d mask that includes all pixel's in strictly before"""
            K = kernel_size
            # mask is DHW
            mask = torch.ones(filter_shape, dtype=torch.float32, requires_grad=False)
            # zero out D=1,
            # - everything to the right of the central pixel, including the central pixel
            mask[-1, K // 2, K // 2:] = 0
            # - all rows below the central row
            mask[-1, K // 2 + 1:, :] = 0

            mask.unsqueeze_(-1).unsqueeze_(-1)  # Make into DHWio, for broadcasting with 3D filters
            return mask

    @staticmethod
    def create_maskB(kernel_size, filter_shape):
        """create 5d mask that includes all pixel's in before and current pixel"""
        K = kernel_size
        # mask is DHW
        mask = torch.ones(filter_shape, dtype=torch.float32, requires_grad=False)
        # zero out D=1,
        # - everything to the right of the central pixel, including the central pixel
        mask[-1, K // 2, K // 2 + 1:] = 0
        # - all rows below the central row
        mask[-1, K // 2 + 1:, :] = 0

        mask.unsqueeze_(-1).unsqueeze_(-1)  # Make into DHWio, for broadcasting with 3D filters
        return mask



class MaskedResblock(nn.Module):
    def __init__(self,channels,filter_shape, kernel_size ):
        super().__init__()


        conv0 = self._create_mask_b_conv(channels,filter_shape, kernel_size)
        conv2 = self._create_mask_b_conv(channels,filter_shape, kernel_size)

        self.model = nn.Sequential(conv0,nn.ReLU(),conv2)

    def forward(self,x):
        return self.model(x) + x[..., 2:, 2:-2, 2:-2, :] # fit the padding

    @staticmethod
    def _create_mask_b_conv(channels,filter_shape, kernel_size ):
        mask = MaskedConv3d.create_maskB(kernel_size, filter_shape)
        return MaskedConv3d(in_channels=channels,out_channels= channels,kernel_size=kernel_size,
                         filter_mask=mask)




class ProbClassifier(nn.Module):
    def __init__(self,classifier_in_channels,classifier_out_channels,kernel_size = 3 ):
        super().__init__()
        self.kernel_size = kernel_size




        K = self.kernel_size
        self.filter_shape = (K // 2 + 1, K, K) # CHW
        CONV0_OUT_CH  = 24


        mask_A = MaskedConv3d.create_maskA(self.kernel_size, self.filter_shape)
        conv0 = MaskedConv3d(in_channels=classifier_in_channels, out_channels=CONV0_OUT_CH,
                             kernel_size =kernel_size,
                             filter_mask=mask_A)

        resblock = MaskedResblock(channels=CONV0_OUT_CH,filter_shape= self.filter_shape,kernel_size= K)

        mask_B = MaskedConv3d.create_maskB(self.kernel_size, self.filter_shape)
        conv2 = MaskedConv3d(in_channels=CONV0_OUT_CH,out_channels= classifier_out_channels,
                         kernel_size =kernel_size,
                         filter_mask=mask_B)

        self.model= nn.Sequential(self.zero_pad_layer(),conv0,nn.ReLU(),resblock,conv2,nn.ReLU())

    def forward(self,x):
        return self.model(x)


    def zero_pad_layer(self):
        """
        :param x: NCHW tensorflow Tensor or numpy array
        """
        nof_conv_layers_classifier = 4 # 4 convd3d layers
        context_size = nof_conv_layers_classifier *(self.kernel_size - 1) + 1
        pad = context_size // 2 # 4

        #padding_left , padding_right , padding_top , padding_bottom , padding_front , padding_back

        pads= (pad,pad,pad,pad,pad,0)

        return nn.ConstantPad3d(pads,value =0)
