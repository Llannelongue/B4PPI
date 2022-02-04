from pytorch_lightning.callbacks import Callback

import torch

import time

import wandb

class Callback_LogRuntimePerEpoch(Callback):

    def __init__(self):
        self.epoch_running_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        self.epoch_running_times.append(time.time()-self.start_time)

    def on_train_end(self, trainer, pl_module):
        n_epochs = trainer.current_epoch+1

        assert len(self.epoch_running_times) == n_epochs

        runtime_per_epoch = (sum(self.epoch_running_times)/n_epochs)/60 # in minutes
        wandb.log({'runtime_per_epoch':runtime_per_epoch})
        # TODO: check that the runtimes per epoch make vaguely sense on full-scale examples.