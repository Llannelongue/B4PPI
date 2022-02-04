import os
import pickle
from argparse import ArgumentParser

import torch
import wandb
import torchmetrics

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

from toolbox_ML import wandb_logging


class PPI_lightning(LightningModule):

    def __init__(self, hparams):
        '''
        This class is the parent class for PPI predediction.
        Can be used by any model.
        :param hparams:
        '''
        super().__init__()

        self.save_hyperparameters(hparams)

        ##################
        # Define metrics #
        ##################

        # self.validation_AP = pl.metrics.AveragePrecision()
        # # self.validation_F1 = pl.metrics.F1(
        # #     average='none',
        # #     compute_on_step = False,
        # # )
        # self.test_AP = pl.metrics.AveragePrecision()
        # self.test_PRC = pl.metrics.PrecisionRecallCurve()
        # self.test_ROC = pl.metrics.ROC()

        self.validation_AP = torchmetrics.AveragePrecision()
        self.test_AP = torchmetrics.AveragePrecision()
        self.test_PRC = torchmetrics.PrecisionRecallCurve()
        self.test_ROC = torchmetrics.ROC()

        self.suffix_testMetrics = '/test'  # can be changed dynamically in lightning_main.py

    def _shared_step(self, batch):
        inputAB, labels = batch

        out = self(inputAB)
        out_probas = self.proba(out)
        labels = labels.type_as(out).view(-1)

        outputs = {
            'out': out,
            'out_probas':out_probas,
            'labels': labels,
            'idA': inputAB['inputA']['IDs'],
            'idB': inputAB['inputB']['IDs'],
        }

        # outputs['loss'] = self.loss_function(outputs)
        return outputs

    # def training_step(self, batch, batch_idx):
    #     outputs = self._shared_step(batch)
    #     loss = self.train_loss_function(outputs)
    #     self.log('loss/train', loss)
    #
    #     return loss

    # FIXME: follow-up on PL bug with float16, using `training_step_end` shouldn't make a difference but it does here. And step_end is more stable.
    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        return outputs

    def training_step_end(self, outputs):
        loss = self.train_loss_function(outputs)
        self.log('loss/train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        loss = self.val_loss_function(outputs)
        self.log('loss/ESeval', loss)

        self.validation_AP(outputs['out_probas'], outputs['labels'])
        self.log('AP/ESeval', self.validation_AP, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)

        self.test_AP(outputs['out_probas'], outputs['labels'])
        self.log(f'AP{self.suffix_testMetrics}', self.test_AP, on_step=False, on_epoch=True)

        return outputs

    def test_epoch_end(self, outputs):

        out_probas = torch.cat([x['out_probas'] for x in outputs])  # These are probabilities (been through a segmoid)
        # out_bin = (out_probas > 0.5).float()
        labels = torch.cat([x['labels'] for x in outputs])

        if not self.hparams.no_logger:
            prc = dict()
            prc['precision'], prc['recall'], _ = self.test_PRC(out_probas, labels)

            roc = dict()
            roc['fpr'], roc['tpr'], _ = self.test_ROC(out_probas, labels)

            wandb_logging(
                args=None,
                prc=prc,
                roc=roc,
                init=False,
                suffix=self.suffix_testMetrics
            )

        # Write out predictions for cross-validation
        if self.hparams.write_out_testPreds:
            print("\nWriting out test results")
            base_path = os.path.join(
                self.hparams.XVal_outputDir,
                f'{self.hparams.wandb_name}_'
            )
            base_path_fold = base_path + f'fold{self.hparams.idx_partition}_'
            path_hparams = base_path + 'hparams.pkl'

            # write out the tensors with the predicted probas and true labels
            torch.save(out_probas, base_path_fold + 'outProbas.pt')
            torch.save(labels, base_path_fold + 'labels.pt')

            # write out the hparams dict to be used by the XVal aggregator
            if not os.path.isfile(path_hparams):
                with open(path_hparams, 'wb') as f:
                    pickle.dump(dict(self.hparams), f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        outputs = self._shared_step(batch)
        return outputs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--optimizer', type=str, default='adam')

        return parser