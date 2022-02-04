from toolbox_DL import *

from lightning_data import PPIDataModule

class PPIDataModule_trainNtest(PPIDataModule):

    def __init__(self, hparams):
        super().__init__(hparams)

    def setup(self):
        self._shared_setup_start()

        ######################
        # Load gold standard #
        ######################

        ids, self.dict_labels, self.dict_mappingID = load_goldStandard(
            args=self.hparams,
            out='IDdict',
            whichSet=[self.hparams.label_train, self.hparams.label_test],
            verbose=False,
        )

        ## Then, load similarity measures data
        if self.hparams.include_similarityMeasures:
            self.GSenriched_df, _, _ = load_goldStandard(
                args=self.hparams,
                out='enrichedDF',
                whichSet=[self.hparams.label_train, self.hparams.label_test],
                verbose=False,
            )
            self.similarityMeasures_inputSize = len(self.hparams.listFeatures2concat)
        else:
            self.GSenriched_df = None
            self.similarityMeasures_inputSize = 0

        ## Then, build partition
        partition_list = [dict(
            train=ids.loc[ids.trainTest == self.hparams.label_train].interactionID.tolist(),
            validation=ids.loc[ids.trainTest == self.hparams.label_test].interactionID.tolist()
        )] # built as a list to fit into the sampler

        if self.hparams.data_sample:
            partition_list = sampleFromXval(
                partition_list,
                train_sampleSize=1001,
                val_sampleSize=301,
                verbose=False
            )

        self.partition = partition_list[0]

        # Create the EarlyStopping evaluation set
        ESsplit = sklearn.model_selection.train_test_split(
            self.partition['train'],
            test_size=0.3,
            random_state=self.hparams.trainValEarlyStop_random_state,
        )
        self.train_IDs = ESsplit[0]
        self.val_IDs = ESsplit[1]
        self.test_IDs = self.partition['validation']

        print()
        print(f"Training on {len(self.train_IDs)+len(self.val_IDs):,} examples (incl. {len(self.val_IDs):,} for validation)")
        print(f"Testing on {len(self.test_IDs):,} examples")
        print()

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # parser.add_argument('--test1', action='store_true') # to remove
        parser.add_argument('--trainNtest', action='store_true')

        return parser