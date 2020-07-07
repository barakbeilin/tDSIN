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
import pdb  

class CustomLoss(nn.Module):
    def forward(self, input, target):
        raise Exception("Shit")


def main2():
    si_autoencoder = SideInformationAutoEncoder(config.use_si_flag)
    loss_manager = LossManager()
    si_net_loss = loss_manager.create_si_net_loss()
    image_list = SideinformationImageImageList.from_csv(
        path="/mnt/code/repos/tDSIN/src/dsin/data", csv_names=["tiny_KITTI_stereo_train.txt", "tiny_KITTI_stereo_val.txt"])
    batchsize = 1
    data = (image_list.split_by_valid_func(lambda x: 'testing' in x)
            .label_from_func(lambda x: x)
            .databunch(bs=batchsize))


    # try:
    my_learner = Learner(data=data,
                            model=si_autoencoder,
                            opt_func=torch.optim.Adam,
                            loss_func=CustomLoss())
    # pdb.set_trace()
    my_learner.fit(1)
    # except Exception as e:
    #     print(e)


if __name__ == '__main__':
    main2()
