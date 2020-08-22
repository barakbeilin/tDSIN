import logging
from fastai import *
from fastai.callbacks import *


def setup_file_logger(log_file='out.log'):
    logger = logging.getLogger()
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    logger.info("start")
    return logger


class BitEntropy(Callback):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self,loss_man,alpha=0.1, logger=logger):
        # If it's a partial, use func.func
        #         name = getattr(func,'func',func).__name__
        self.loss_man = loss_man
        self.alpha = alpha
        self.logger = logger
        
    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val = 0.0
        self.iter = 0
        
        self.val_si_loss = None
        
    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        
        self.val *= 1 - self.alpha
        self.val += self.alpha * self.loss_man.soft_bit_entropy.detach()
        if self.val_si_loss is None:
            self.val_si_loss = self.loss_man.si_net_loss_value
        else :
            self.val_si_loss *= 1 - self.alpha
            self.val_si_loss += self.alpha * self.loss_man.si_net_loss_value.detach()
            
        if self.iter % 500 == 0:
            importance_map = self.loss_man.importnace_map_dict
            msg = f"iter {self.iter}: bpp = {self.val / 2:.3f} "
            msg +=f"imp-mean-var({importance_map["mean"]:.2f} {importance_map["var"]:.2f})"
            msg += f" total loss{self.loss_man.total_loss:.1f}  l2reg_loss={self.loss_man.l2_reg_loss:.1f}"
            msg += f"autoencoder_loss_value={ self.loss_man.autoencoder_loss_value:.1f}"
            msg += f"si_loss={self.val_si_loss:.1f}"
            msg += f"feat_loss_value={self.loss_man.feat_loss_value:.1f}"
            self.logger.info(msg)
            print(msg)
        self.iter += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val)