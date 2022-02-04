
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

from lightning_model import PPI_lightning

class OmicsRegression_lightning(PPI_lightning):

    def __init__(self, hparams,
                 # concat_inputSize, similarityMeasures_inputSize
                 ):
        '''
        Simple logistic regression on long form omics features.
        :param hparams:
        :param concat_inputSize:
        '''
        super().__init__(hparams)

        if self.hparams.include_similarityMeasures:
            self.inputSize = self.hparams.similarityMeasures_inputSize
        else:
            self.inputSize = self.hparams.concat_inputSize*2

        self.linear = nn.Linear(
            in_features=self.inputSize,
            out_features=1,
            bias=not self.hparams.no_bias,
        )

        self.proba = nn.Sigmoid()

    def configure_optimizers(self):
        assert self.hparams.optimizer == 'adam'

        if self.hparams.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_loss_function(self, outputs):
        loss = F.binary_cross_entropy_with_logits(
            outputs['out'],
            outputs['labels'],
            reduction='sum',
        )

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg

        loss /= outputs['out'].size(0)

        return loss

    def val_loss_function(self, outputs):
        return F.binary_cross_entropy_with_logits(
            outputs['out'],
            outputs['labels'],
            reduction='mean',
        )

    def forward(self, inputAB):
        '''
        Forward pass
        :param inputAB: {'inputA':{'Xconcat','Xseq','length'}, 'inputB':{'Xconcat','Xseq','length'}}
        :return:
        '''
        if self.hparams.include_similarityMeasures:
            inputConcat = inputAB['similarityMeasuresAB']
        else:
            inputConcat = torch.cat([inputAB['inputA']['Xconcat'], inputAB['inputB']['Xconcat']], dim=1)
        assert inputConcat.size(1) == self.inputSize

        y_hat = self.linear(inputConcat).view(-1)

        return y_hat


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--no_bias', action='store_true')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--l1_strength', type=float, default=0.0)
        parser.add_argument('--l2_strength', type=float, default=0.0)

        return parser