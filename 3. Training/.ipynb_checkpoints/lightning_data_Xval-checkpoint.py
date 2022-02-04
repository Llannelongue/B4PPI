from toolbox_DL import *

from lightning_data import PPIDataModule

class PPIDataModule_Xval(PPIDataModule):

    def __init__(self, hparams):
        super().__init__(hparams)

    def setup(self):
        self._shared_setup_start()

        ######################
        # Load gold standard #
        ######################

        idsGS, self.dict_labels, self.dict_mappingID = load_goldStandard(
            args=self.hparams,
            out='IDdict',
            whichSet=[self.hparams.label_train],
            verbose=False,
        )

        if self.hparams.include_similarityMeasures:
            self.GSenriched_df, _, _ = load_goldStandard(
                args=self.hparams,
                out='enrichedDF',
                whichSet=[self.hparams.label_train],
                verbose=False,
            )
            self.similarityMeasures_inputSize = len(self.hparams.listFeatures2concat)
        else:
            self.GSenriched_df = None
            self.similarityMeasures_inputSize = 0

        ###################
        # Split train/val #
        ###################

        # List of dicts {'train':[]. 'validation':[]}
        cv_partition = stratifiedXvalPartitions(
            listIDs=idsGS.interactionID.to_list(),
            listLabels=idsGS.isInteraction.to_list(),
            IDorIndex='ID',
            n_splits=self.hparams.Xval_n_folds,
            random_state=self.hparams.Xval_random_state,
            verbose=False,
        )

        if self.hparams.data_sample:
            cv_partition = sampleFromXval(
                cv_partition,
                train_sampleSize=1001,
                val_sampleSize=301,
                verbose=False
            )

        self.partition = cv_partition[self.hparams.idx_partition]

        ESsplit = sklearn.model_selection.train_test_split(
            self.partition['train'],
            test_size=0.3,
            random_state=self.hparams.trainValEarlyStop_random_state,
        )
        self.train_IDs = ESsplit[0]
        self.val_IDs = ESsplit[1]
        self.test_IDs = self.partition['validation']


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        #### Xval ####
        parser.add_argument('--Xval', action='store_true')
        parser.add_argument('--Xval_n_folds', type=int, default=4)
        parser.add_argument('--Xval_random_state', type=int, default=876)
        parser.add_argument('--idx_partition', type=int, default=0)

        return parser
