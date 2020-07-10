from fastai import *
from fastai.vision import *
import torch
from torch import nn
from dsin.ae.data_manager.data_loader import (
    SideinformationImageImageList, ImageSiTuple)
from dsin.ae import config
from dsin.ae.si_ae import SideInformationAutoEncoder
from dsin.ae.si_net import SiNetChannelIn
from dsin.ae.loss_man import LossManager
from dsin.ae.distortions import Distortions
from dsin.ae.kitti_normalizer import ChangeImageStatsToKitti, ChangeState
from dsin.ae import config


class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."

    def __init__(self, func):
        # If it's a partial, use func.func
        name = getattr(func, 'func', func).__name__
        self.func, self.name = func, name

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0., 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target):
            last_target = [last_target]
        self.count += last_target[0].size(0)  # batch size
        X_DEC_IND = 1
        val = self.func(last_output[X_DEC_IND], last_target[0])
        self.val += last_target[0].size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)


def main():
    si_autoencoder = SideInformationAutoEncoder(config.use_si_flag)
    image_list = SideinformationImageImageList.from_csv(
        path="../src/dsin/data", csv_names=["tiny_KITTI_stereo_train.txt", "tiny_KITTI_stereo_val.txt"])
    ll = image_list.split_by_valid_func(
        lambda x: 'testing' in x).label_from_func(lambda x: x)
    batchsize = 1
    data = (image_list
            .split_by_valid_func(lambda x: 'testing' in x)
            .label_from_func(lambda x: x)
            .transform(None, size=(336, 1224), resize_method=ResizeMethod.CROP, tfm_y=True)
            .databunch(bs=batchsize))
    my_learner = Learner(data=data,
                         model=si_autoencoder,
                         opt_func=torch.optim.Adam,
                         loss_func=LossManager(config.use_si_flag),
                         metrics=[AverageMetric(Distortions._calc_dist)])
    my_learner.load('tiny-1')
    datum = my_learner.data.train_ds[0][0]
    
    
    # my_learner.predict(datum)
    my_learner.show_results()


if __name__ == "__main__":
    main()
