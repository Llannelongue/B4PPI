
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

from lightning_buildingBlocks import *

from lightning_model import PPI_lightning


class SiameseHybrid_lightning(PPI_lightning):

    def __init__(self, hparams,
                 # concat_inputSize, seq_inputSize, similarityMeasures_inputSize
                 ):
        '''
        Siamese-based prediction models.
        :param hparams:
        :param concat_inputSize:
        :param seq_inputSize:
        '''
        super().__init__(hparams)

        self.concat_inputSize = self.hparams.concat_inputSize
        self.seq_inputSize = self.hparams.seq_inputSize

        # Sanity checks
        assert self.hparams.hybridType in ["omics", 'hybrid', 'sequence']

        ##############################
        # Initialise building blocks #
        ##############################

        dimOutEmbedding = 0

        if self.hparams.hybridType in ["omics", 'hybrid']:
            omicsEmbedding = omicsEmbedding_MLP(
                # concat_inputSize=self.concat_inputSize,
                hparams=self.hparams,
            )
            self.omicsEmbedding = omicsEmbedding.embedding
            dimOutEmbedding += omicsEmbedding.dimOut

        if self.hparams.hybridType in ["sequence", 'hybrid']:
            sequenceEmbedding = sequenceEmbedding_RNN(
                # seq_inputSize=self.seq_inputSize,
                hparams=self.hparams,
            )
            self.rnn = sequenceEmbedding.rnn
            self.sequenceEmbedding = sequenceEmbedding.embedding
            dimOutEmbedding += sequenceEmbedding.dimOut

        dimOutEmbedding += self.hparams.similarityMeasures_inputSize

        self.out = siameseOutput(
            hparams=self.hparams,
            dimOutEmbedding=dimOutEmbedding,
        ).out

        self.proba = nn.Sigmoid()

    def configure_optimizers(self):

        assert self.hparams.optimizer == 'adam'

        params_listDict = [
            {'params': self.out.parameters(), 'lr': self.hparams.out_lr},
            {'params': self.proba.parameters(), 'lr': self.hparams.out_lr},
        ]

        if self.hparams.hybridType in ["omics", 'hybrid']:
            params_listDict.append({
                'params': self.omicsEmbedding.parameters(),
                'lr': self.hparams.concat_lr
            })

        if self.hparams.hybridType in ["sequence", 'hybrid']:
            params_listDict += [
                {'params': self.rnn.parameters(), 'lr': self.hparams.seq_lr},
                {'params': self.sequenceEmbedding.parameters(), 'lr': self.hparams.seq_postRNN_lr}
            ]

        if self.hparams.optimizer == 'adam':
            return torch.optim.Adam(params_listDict)

    def train_loss_function(self, outputs):
        # BCEwithLogitLoss requires raw outputs (not been through a sigmoid yet)
        return F.binary_cross_entropy_with_logits(
            outputs['out'],
            outputs['labels'],
            reduction='mean',
        )

    def val_loss_function(self, outputs):
        return self.train_loss_function(outputs)

    def forward_one(self, inputX):
        '''
        Forward pass for one half of the network
        :param inputX: dict with keys ['Xconcat','Xseq','length']
        :return:
        '''

        if self.hparams.hybridType in ["omics", "hybrid"]:
            out_omics = self.omicsEmbedding(inputX['Xconcat'])

        if self.hparams.hybridType in ["sequence", "hybrid"]:
            # we need to pack the sequence to deal with sequences of different lengths
            Xseq_pack = torch.nn.utils.rnn.pack_padded_sequence(
                inputX['Xseq'],
                inputX['Xlengths'].cpu(), # .cpu() to deal with PyTorch issue https://github.com/pytorch/pytorch/issues/43227
                batch_first=True,
                enforce_sorted=False
            )

            # Go through the RNN
            if self.hparams.seq_RNN == 'LSTM':
                outRNN_pack, (ht, ct) = self.rnn(Xseq_pack)
            elif self.hparams.seq_RNN == 'GRU':
                outRNN_pack, ht = self.rnn(Xseq_pack)

            # Select the correct output and reshape it
            if self.hparams.seq_bidirectional:
                # output are the last 2 hidden states
                # permute switches to `batch_first`, then select last 2 hidden states
                # and then reshape concat the different layers' outputs
                out_sequence = ht.permute(1, 0, 2)[:, ht.size(0) - 2:, :].reshape([ht.size(1), -1])
            else:
                # output is last hidden state
                out_sequence = ht[-1]

            # Pass through the post-RNN embedding
            # Possibly transparent if no embedding is specified
            out_sequence = self.sequenceEmbedding(out_sequence)

        if self.hparams.hybridType == "omics":
            return out_omics
        elif self.hparams.hybridType == "sequence":
            return out_sequence
        else:
            return torch.cat((out_omics, out_sequence), dim=1)

    def forward(self, inputAB):
        # inputAB: dict with keys ['inputA', 'inputB']
        # Each 'inputX' is also a dict with keys ['Xconcat','Xseq','length']
        # Global forward pass

        outA = self.forward_one(inputAB['inputA'])
        outB = self.forward_one(inputAB['inputB'])
        dis = torch.abs(outA - outB)

        if self.hparams.include_similarityMeasures:
            dis = torch.cat([dis, inputAB['similarityMeasuresAB'].type_as(dis)], dim=1)

        out = self.out(dis).view(-1)

        return (out)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        #### hparam ####
        hybridType = 'hybrid'
        parser.add_argument('--hybridType', type=str, default=hybridType)  # 'hybrid', 'sequence' or 'omics'
        parser.add_argument('--include_similarityMeasures', action='store_true')

        parser.add_argument('--concat_sizeLayers', type=str, default='250,200')
        parser.add_argument('--concat_sizeEmbedding', type=int, default=128)
        parser.add_argument('--concat_pDropout', type=float, default=0.3)
        parser.add_argument('--concat_lr', type=float, default=1e-4)

        parser.add_argument('--seq_sizeHidden', type=int, default=10)
        parser.add_argument('--seq_RNN', type=str, default='GRU')
        parser.add_argument('--seq_bidirectional', dest='seq_bidirectional', action='store_true')
        parser.add_argument('--seq_monodirectional', dest='seq_bidirectional', action='store_false')
        parser.set_defaults(seq_bidirectional=True)
        parser.add_argument('--seq_n_layers', type=int, default=1)
        parser.add_argument('--seq_pDropout', type=float, default=0)
        parser.add_argument('--seq_lr', type=float, default=1e-3)
        parser.add_argument('--seq_postRNN', type=str, default='none')  # ['none','fc']
        parser.add_argument('--seq_postRNN_sizeLayers', type=str, default='')  #string of int such as '50,20'
        parser.add_argument('--seq_sizeEmbedding', type=int, default=128)
        parser.add_argument('--seq_postRNN_pDropout', type=float, default=0.3)
        parser.add_argument('--seq_postRNN_lr', type=float, default=1e-4)

        parser.add_argument('--out_type', type=str, default='linear')  # 'linear' or 'dense'
        parser.add_argument('--out_sizeLayers', type=str, default='25')
        parser.add_argument('--out_lr', type=float, default=1e-4)
        parser.add_argument('--out_pDropout', type=float, default=0.3)

        return parser

