import torch
import torch.nn.functional as F
from torch import nn
from enum import Enum, auto
def pad_for_probclass3d(x, context_size, pad_value=0, learn_pad_var=False):
    """
    :param x: NCHW tensorflow Tensor or numpy array
    """
    input_is_tf = not isinstance(x, np.ndarray)
    if not input_is_tf and x.ndim == 3:  # for bit_counter
        return remove_batch_dim(pad_for_probclass3d(
                add_batch_dim(x), context_size, pad_value, learn_pad_var))

    with tf.name_scope('pad_cs' + str(context_size)):
        pad = context_size // 2
        assert pad >= 1
        if learn_pad_var:
            if not isinstance(pad_value, tf.Variable):
                print('Warn: Expected tf.Variable for padding, got {}'.format(pad_value))
            return pc_pad_grad(x, pad, pad_value)

        pads = [[0, 0],  # don't pad batch dimension
                [pad, 0],  # don't pad depth_future, it's not seen by any filter
                [pad, pad],
                [pad, pad]]
        assert len(pads) == _get_ndims(x), '{} != {}'.format(len(pads), x.shape)

        pad_fn = tf.pad if input_is_tf else get_np_pad_fn()
        return pad_fn(x, pads, constant_values=pad_value)
