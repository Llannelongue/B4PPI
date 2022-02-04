from toolbox_ML import *

import time

import torch
from pytorch_lightning import Trainer

from lightning_model_siamese import SiameseHybrid_lightning
from lightning_model_regression import OmicsRegression_lightning

from lightning_data_predict import PPIDataModule_predict


def export_predictions(preds,dm,args):
    # Load help files
    cfg = load_cfg(path2dir='../..')

    file_out = os.path.join(
        cfg['outputPredictedPPI'],
        f"predictions__{args.modelID}__{dm.nameDF}.pkl"
    )

    if os.path.exists(file_out):
        print("Overwritting old predictions")

    preds.to_pickle(file_out)
    print(f"Predictions exported: {file_out}")

def main_predict(args):

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    ### Load model from checkpoint
    if args.modelID == '':
        assert args.path2model != ''
    else:
        path0 = os.path.join(
            cfg['modelsCheckpoints'],
            args.modelID
        )

        L = []
        for path, subdirs, files in os.walk(path0):
            for name in files:
                L.append(os.path.join(path, name))

        assert len(L) == 1
        args.path2model = L[0]

    checkpoint = torch.load(args.path2model, map_location=device)
    hp_checkpoint = checkpoint['hyper_parameters']

    # FIXME: this workaround is only needed for the models trained before 04/06/2021
    if 'concat_inputSize' not in hp_checkpoint:
        hp_checkpoint['concat_inputSize'] = 22941
        hp_checkpoint['seq_inputSize'] = 21
        if hp_checkpoint['include_similarityMeasures']:
            hp_checkpoint['similarityMeasures_inputSize'] = 10
        else:
            hp_checkpoint['similarityMeasures_inputSize'] = 0

    if hp_checkpoint['model_family'] == 'Siamese':
        model = SiameseHybrid_lightning.load_from_checkpoint(
            args.path2model,
            **hp_checkpoint
        )
    elif hp_checkpoint['model_family'] == 'OmicsRegression':
        model = OmicsRegression_lightning.load_from_checkpoint(
            args.path2model,
            **hp_checkpoint
        )
    else:
        model = None

    # model.hparams.species = args.species
    hp_checkpoint['species'] = args.species

    ### Initialise data module
    dm = PPIDataModule_predict(hparams=dotdict(hp_checkpoint), args=args)
    dm.setup()

    ### Trainer
    trainer = Trainer.from_argparse_args(args)

    ### Make predictions
    predictions = trainer.predict(
        model=model,
        datamodule=dm
    )

    # Move predictions to cpu
    predictions2 = []
    for x in predictions:
        x2 = dict()
        for key, value in x.items():
            try:
                x2[key] = value.cpu()
            except:
                x2[key] = value
        predictions2.append(x2)

    # Concat predictions
    predDF = pd.concat([pd.DataFrame(x) for x in predictions2], axis=0, join='outer')

    # Export predictions
    export_predictions(predDF, dm, args)

if __name__ == '__main__':

    cfg = load_cfg(path2dir='../..')
    logVersions = load_LogVersions('../..')
    mapping_modelIDs = load_modelIDs()

    #################
    # CREATE PARSER #
    #################

    parser = ArgumentParser()

    parser.add_argument('--predict', action='store_true')

    ## We only use path2model if runID == ''
    parser.add_argument('--modelID', type=str, default='')
    parser.add_argument('--path2model', type=str, default='')
    # parser.add_argument('--modelName', type=str, default='')
    # parser.add_argument('--model_family', type=str, default='Siamese')  # [Siamese, OmicsRegression]

    parser = Trainer.add_argparse_args(parser)
    parser = PPIDataModule_predict.add_data_specific_args(parser)

    ##################
    # PULL ARGUMENTS #
    ##################

    args = parser.parse_args()

    #######################
    # ADD FIXED ARGUMENTS #
    #######################

    args.cpus_per_gpu = 32

    if isinstance(args.gpus, int):
        args.num_workers = max(1, args.gpus * args.cpus_per_gpu)
    else:
        args.num_workers = 1

    # Test
    # args.modelID = '162383'
    # args.path2model = "/rds/user/ll582/hpc-work/PPIdata/output/4_predictions/training_logs/modelsCheckpoints/162383/epoch=2.ckpt" # fyfsyudp


    #############
    # RUN MODEL #
    #############

    for modelName in [
        'SiamSeq_y-01'
        # 'SiamSeq_b01'
    ]:
        for newData in [
            'benchmarkingGS_test1_v1-0',
            'benchmarkingGS_test2_v1-0',
            # 'benchmarkingGS_4hubs_test_v1-0',
            # 'benchmarkingGS_yeast_test_v1-0',
            # 'benchmarkingGS_yeast_4hubs_test_v1-0'
        ]:
            args.species = 'human'
            # args.species = 'yeast'

            print(modelName)
            print(newData)
            print()

            ID = mapping_modelIDs[modelName]
            args.newData = newData
            args.modelID = ID
            main_predict(args)
            print()