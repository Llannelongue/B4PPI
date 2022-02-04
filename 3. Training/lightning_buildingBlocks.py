import torch

class MLP_generator(torch.nn.Module):
    def __init__(
            self,
            size_in, size_out, size_hidden=None,
            p_dropout=0.,
            prefixLayer='',
            keep_last_linear=False,
    ):
        '''
        Create a nn.Sequential module of a MLP of arbitrary length
        :param size_in: int
        :param size_hidden: list of int
        :param size_out: int
        '''
        super().__init__()

        if size_hidden is None:
            size_hidden = []

        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        self.p_dropout = p_dropout
        self.prefixLayer = prefixLayer
        self.keep_last_linear=keep_last_linear

    def create_MLP(self):

        MLP = torch.nn.Sequential()
        size_in = self.size_in

        for i, dimension in enumerate(self.size_hidden + [self.size_out]):
            module = torch.nn.Sequential()
            module.add_module('fc',torch.nn.Linear(size_in, dimension))

            if (i < len(self.size_hidden))|(not self.keep_last_linear):
                # means we are not in the last layer
                module.add_module('ReLu',torch.nn.ReLU())
                module.add_module('dropout',torch.nn.Dropout(self.p_dropout))

            MLP.add_module(f'{self.prefixLayer}fc{i}', module)
            size_in = dimension

        return MLP


class omicsEmbedding_MLP(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()

        # Create list of dimensions of hidden layers
        sizeLayers = [int(x) for x in hparams.concat_sizeLayers.split(',') if x]

        # Sanity checks
        assert 0 not in sizeLayers

        ### Build fully connected embedding
        self.embedding = MLP_generator(
            size_in = hparams.concat_inputSize,
            size_out = hparams.concat_sizeEmbedding,
            size_hidden=sizeLayers,
            p_dropout=hparams.concat_pDropout,
            prefixLayer='omics_'
        ).create_MLP()

        self.dimOut = hparams.concat_sizeEmbedding


class sequenceEmbedding_RNN(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()

        ### Build RNN embedding (many-to-one)

        assert hparams.seq_RNN in ['LSTM','GRU']

        if hparams.seq_RNN == 'LSTM':
            self.rnn = torch.nn.LSTM(
                input_size=hparams.seq_inputSize,
                hidden_size=hparams.seq_sizeHidden,
                batch_first=True,
            )

        elif hparams.seq_RNN == 'GRU':
            self.rnn = torch.nn.GRU(
                input_size=hparams.seq_inputSize,
                hidden_size=hparams.seq_sizeHidden,
                batch_first=True,
                bidirectional=hparams.seq_bidirectional,
                num_layers=hparams.seq_n_layers,
                dropout=hparams.seq_pDropout,
            )

        self.dimOut = hparams.seq_sizeHidden * (1 + int(hparams.seq_bidirectional))

        # Post-RNN processing (potentially transparent)
        assert  hparams.seq_postRNN in ['none','fc']
        if hparams.seq_postRNN == 'none':
            self.embedding = torch.nn.Sequential()
        elif hparams.seq_postRNN == 'fc':
            sizeLayers = [int(x) for x in hparams.seq_postRNN_sizeLayers.split(',') if x]
            assert 0 not in sizeLayers

            self.embedding = MLP_generator(
                size_in=self.dimOut,
                size_out=hparams.seq_sizeEmbedding,
                size_hidden=sizeLayers,
                p_dropout=hparams.seq_postRNN_pDropout,
                prefixLayer='seq_postRNN_',
                keep_last_linear=False
            ).create_MLP()

            self.dimOut = hparams.seq_sizeEmbedding


class siameseOutput(torch.nn.Module):

    def __init__(self, hparams, dimOutEmbedding):
        super().__init__()
        assert hparams.out_type in ['linear','dense']

        if hparams.out_type == 'linear':
            sizeLayers = None
        else:
            # Create list of dimensions of hidden layers
            sizeLayers = [int(x) for x in hparams.out_sizeLayers.split(',') if x]

        self.out = MLP_generator(
            size_in=dimOutEmbedding,
            size_out=1,
            size_hidden=sizeLayers,
            p_dropout=hparams.out_pDropout,
            prefixLayer='out_',
            keep_last_linear=True
        ).create_MLP()


if __name__ == '__main__':
    # Used for testing
    foo = MLP_generator(
        size_in = 24,
        size_out=1,
        # size_hidden=[30,32,34],
        size_hidden=None,
        p_dropout=0.1,
        prefixLayer='bla_',
        keep_last_linear=True
    ).create_MLP()

    print(foo)

    x = torch.randn(100, 24)

    y = foo(x)

    print(y.shape)