from toolbox_DL import *

from lightning_data import PPIDataModule

class PPIDataModule_predict(PPIDataModule):
    def __init__(self, hparams, args):
        '''
        :param hparams: from the pre-trained model
        :param args:  from the command line, to control prediction
        '''''
        super().__init__(hparams)
        self.args = args

    def setup(self, stage = None):
        self._shared_setup_start()

        ################
        # Load dataset #
        ################

        nameDF = self.args.newData

        ids, self.dict_labels, self.dict_mappingID = load_data4prediction(
            nameDF=nameDF,
            out='IDdict',
        )
        self.ids2predict = ids.interactionID.tolist()

        ## Then, load similarity measures data
        if self.hparams.species == 'human':
            DFversion = self.logVersions['featuresEngineering']['similarityMeasure']
        else:
            DFversion = self.logVersions['featuresEngineering'][self.hparams.species]['similarityMeasure']

        if self.hparams.include_similarityMeasures:
            nameDF += f"_similarityMeasure_v{DFversion}"
            self.dfEnriched, _, _ = load_data4prediction(
                nameDF=nameDF,
                out='enrichedDF',
            )
        else:
            self.dfEnriched = None

        self.nameDF = nameDF

    def predict_dataloader(self):
        predict_dataset = myDataset(
            list_IDs=self.ids2predict,
            mappingID=self.dict_mappingID,
            labels=self.dict_labels,
            data=self.dict_data,
            listFeatures2concat=self.hparams.listFeatures2concat,
            similarityMeasures=self.dfEnriched
        )

        predict_generator = torchData.DataLoader(
            predict_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=PadSequence(),
            pin_memory=True
        )

        return predict_generator


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--newData', type=str, default='')

        parser.add_argument('--batch_size', type=int, default=50)

        return parser