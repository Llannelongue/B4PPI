import pytorch_lightning as pl

from toolbox_DL import *

class PPIDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()

        self.cfg = load_cfg(path2dir='../..')
        self.logVersions = load_LogVersions('../..')

        self.hparams = hparams

        if self.hparams.features2concat == '':
            self.hparams.listFeatures2concat = []
        else:
            self.hparams.listFeatures2concat = self.hparams.features2concat.split(',')
        self.hparams.nonNumericFeatures = ['sequence']

    def _shared_setup_start(self):

        ########################
        # Load normalised data #
        ########################

        self.dict_data, self.inputSizeConcat = createDataDict(
            cfg=self.cfg,
            logVersions=self.logVersions,
            hparams=self.hparams,
            verbose=False,
        )

        _, self.seq_inputSize = embeddingChar('C')

    # def _shared_setup_end(self):

    def train_dataloader(self):
        '''
        Create the dataloader for training (i.e. using back-propagation.
        To avoid data leakage between the XVal folds, this needs to be a subset
        of the `train` in the Xval partition (to save some observations for early stopping).
        :return:
        '''

        train_dataset = myDataset(
            list_IDs=self.train_IDs,
            mappingID=self.dict_mappingID,
            labels=self.dict_labels,
            data=self.dict_data,
            listFeatures2concat=self.hparams.listFeatures2concat,
            similarityMeasures=self.GSenriched_df
        )

        train_generator = torchData.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size_loaders,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=PadSequence(),
            pin_memory=True
        )

        return train_generator

    def val_dataloader(self):

        val_dataset = myDataset(
            list_IDs=self.val_IDs,
            mappingID=self.dict_mappingID,
            labels=self.dict_labels,
            data=self.dict_data,
            listFeatures2concat=self.hparams.listFeatures2concat,
            similarityMeasures=self.GSenriched_df
        )

        val_generator = torchData.DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size_loaders,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=PadSequence(),
            pin_memory=True
        )

        return val_generator

    def test_dataloader(self):

        test_dataset = myDataset(
            list_IDs=self.test_IDs,
            mappingID=self.dict_mappingID,
            labels=self.dict_labels,
            data=self.dict_data,
            listFeatures2concat=self.hparams.listFeatures2concat,
            similarityMeasures=self.GSenriched_df
        )

        test_generator = torchData.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size_loaders,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=PadSequence(),
            pin_memory=True
        )

        return test_generator

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        #### Features ####
        features2concat = 'bioProcessUniprot,cellCompUniprot,molFuncUniprot,domainUniprot,' \
                          'motifUniprot,Bgee,tissueCellHPA,tissueHPA,RNAseqHPA,' \
                          'subcellularLocationHPA'
        parser.add_argument('--features2concat', type=str, default=features2concat)
        parser.add_argument('--normaliseBinary', action='store_true')

        #### DataLoaders ####
        parser.add_argument('--batch_size', type=int, default=50)

        parser.add_argument('--data_sample', action='store_true')
        parser.add_argument('--trainValEarlyStop_random_state', type=int, default=595)

        ## labels in GS
        parser.add_argument('--label_train', type=str, default='train')
        parser.add_argument('--label_test', type=str, default='test')

        return parser