from fastai import *
from fastai.vision import *

from fastai.callback import Callback



class BitEntropy(Callback):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self,loss_man,logger,use_si,alpha=0.1):
        # If it's a partial, use func.func
        #         name = getattr(func,'func',func).__name__
        self.loss_man = loss_man
        self.alpha = alpha
        self.logger = logger
        self.use_si = use_si
        
    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val = 0.0
        self.iter = 0
        
        self.val_si_loss = 0
        self.val_ae_loss = 0
    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        
        self.val *= 1 - self.alpha
        self.val += self.alpha * self.loss_man.soft_bit_entropy.detach()
        
        if self.use_si:
            self.val_ae_loss = self.loss_man.autoencoder_loss_value
            
            if self.val_si_loss == 0:
                self.val_si_loss = self.loss_man.si_net_loss_value
            else :
                self.val_si_loss *= 1 - self.alpha
                self.val_si_loss += self.alpha * self.loss_man.si_net_loss_value.detach()
        else:
            if self.val_ae_loss == 0:
                self.val_ae_loss = self.loss_man.autoencoder_loss_value.detach()
            else :
                self.val_ae_loss *= 1 - self.alpha
                self.val_ae_loss += self.alpha * self.loss_man.autoencoder_loss_value.detach()

            self.val_si_loss = self.loss_man.si_net_loss_value
             
            
        if self.iter % 500 == 0:
            importance_map = self.loss_man.importnace_map_dict
            msg = f"iter {self.iter}: bpp = {self.val / 2:.3f} "
            msg +=f"imp-mean-var({importance_map['mean']:.2f} {importance_map['var']:.2f})"
            msg += f" total loss{self.loss_man.total_loss:.1f}  l2reg_loss={self.loss_man.l2_reg_loss:.1f}"
            msg += f"autoencoder_loss_value={ self.val_ae_loss:.1f}"
            msg += f"si_loss={self.val_si_loss:.1f}"
            msg += f"feat_loss_value={self.loss_man.feat_loss_value:.1f}"
            self.logger.info(msg)
            print(msg)
        self.iter += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val)

class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self, func,name):
        # If it's a partial, use func.func
        #         name = getattr(func,'func',func).__name__
        self.func, self.name = func, name

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not isinstance(last_target, (list,tuple)): last_target=[last_target]
        self.count += last_target[0].size(0) # batch size
        val = self.func(last_output, last_target[0])
        self.val += last_target[0].size(0) * val.detach().cpu()
        
    
    
    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)
    
class ParameterMetricCallback(Callback):
    def __init__(self,loss_man):
        self.loss_man = loss_man
    
    def on_backward_begin(self,*args, **kwargs):
        self.pbar=kwargs["pbar"]
        if hasattr(self.loss_man,'soft_bit_entropy'):
            self.pbar.child.comment += f' soft_bit_entropy: {self.loss_man.soft_bit_entropy:.4f}'
            
            
class ProgressBarAttibuteVisualizer(Callback):
    def __init__(self,learner, loss_man,attrs):
        self.loss_man = loss_man
        self.attrs = attrs
    def on_backward_begin(self,*args, **kwargs):
        self.pbar=kwargs["pbar"]
        msg = ""
        
        for attr in self.attrs:
            msg += f"{attr}={getattr(self.loss_man,attr,None):.2f} "
        self.pbar.child.comment += msg
        
        
        
         
class ParameterRunningAverageMetricCallback(Callback):
    def __init__(self,loss_man,alpha=0.1):
        self.loss_man = loss_man
        self.alpha = alpha
        self.val = None
    
    def on_backward_begin(self,*args, **kwargs):
        self.pbar=kwargs["pbar"]
        self.importance_map=self.loss_man.importnace_map_dict

        if hasattr(self.loss_man,'soft_bit_entropy'):
            if self.val is None:
                self.val = self.loss_man.soft_bit_entropy.detach()
            else:
                self.val *= 1 - self.alpha
                self.val += self.alpha * self.loss_man.soft_bit_entropy.detach()
                
            self.pbar.child.comment += f' avg_bpp: {self.val / 2 :.4f} imp-mean-var({self.importance_map["mean"]:.2f} {self.importance_map["var"]:.2f})'
            msg = f"bitcost_loss={self.loss_man.bit_cost_loss_value:.1f} "
            msg += f"qcent_loss={self.loss_man.centers_regularization_term:.1f} "
            msg += f"l2reg_loss={self.loss_man.l2_reg_loss:.1f} "
            msg += f"autoencoder_loss_value={ self.loss_man.autoencoder_loss_value:.1f} "
            msg += f"si_loss={self.loss_man.si_net_loss_value:.1f} "
            msg += f"feat_loss_value={self.loss_man.feat_loss_value:.1f}"

            self.pbar.child.comment += msg
            
