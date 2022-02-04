# This is sort of lightning_main.py v2 (that should handle both Xval and single fold run)

from toolbox_ML import *

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GPUStatsMonitor
import wandb

from argparse import ArgumentParser

import random
import glob

from lightning_data import PPIDataModule
from lightning_data_Xval import PPIDataModule_Xval
from lightning_data_trainNtest import PPIDataModule_trainNtest

from lightning_model import PPI_lightning
from lightning_model_siamese import SiameseHybrid_lightning
from lightning_model_regression import OmicsRegression_lightning

from lightning_callbacks import Callback_LogRuntimePerEpoch


def main(args):
    ################################
    # set seed for reproducibility #
    ################################

    seed_everything(args.torch_seed)

    ####################
    # UPDATE ARGUMENTS #
    ####################

    # Adjust real batch size to pass on to DataLoaders
    if isinstance(args.gpus, int):
        # ie we are working with GPU(s)
        args.batch_size_loaders = args.batch_size * args.gpus
    else:
        args.batch_size_loaders = args.batch_size

    if args.myAmp not in [16,32]:
        if isinstance(args.gpus, int):
            args.myAmp = 16
        else:
            args.myAmp = 32

    # Remove the omics features to concat for sequence
    if (args.hybridType == 'sequence')&(not args.include_similarityMeasures):
        args.features2concat = ''

    ###########################
    # INITIALISE DATA MODULES #
    ###########################

    assert not (args.Xval and args.trainNtest)
    if args.Xval:
        dm = PPIDataModule_Xval(args)
    elif args.trainNtest:
        dm = PPIDataModule_trainNtest(args)
    else:
        dm = PPIDataModule_Xval(args)
    dm.setup()

    ####################
    # INITIALISE MODEL #
    ####################

    assert args.model_family in ['Siamese', 'OmicsRegression']
    args.concat_inputSize = dm.inputSizeConcat
    args.seq_inputSize = dm.seq_inputSize
    args.similarityMeasures_inputSize = dm.similarityMeasures_inputSize
    if args.model_family == 'Siamese':
        model = SiameseHybrid_lightning(
            hparams=args,
            # concat_inputSize=dm.inputSizeConcat,
            # seq_inputSize=dm.seq_inputSize,
            # similarityMeasures_inputSize=dm.similarityMeasures_inputSize,
        )
    elif args.model_family == 'OmicsRegression':
        model = OmicsRegression_lightning(
            hparams=args,
            # concat_inputSize=dm.inputSizeConcat,
            # similarityMeasures_inputSize=dm.similarityMeasures_inputSize,
        )
    else:
        model = None

    #################
    # Create logger #
    #################

    if not args.no_logger:

        args_logger = {
            'save_dir': args.wandbLogs_dir,
            'project': args.wandb_project,
        }

        if args.Xval:
            used_name = f'{args.wandb_name}-fold{args.idx_partition}'
            additional_tags = ['Xval-fold']
        elif args.trainNtest:
            used_name = args.wandb_name
            additional_tags = [args.label_test]
        else:
            used_name = args.wandb_name
            additional_tags = []

        if args.wandb_tags != '':
            args_logger['tags'] = args.wandb_tags.split(',') + additional_tags
        elif len(additional_tags) > 0:
            args_logger['tags'] = additional_tags

        if args.wandb_name != '':
            args_logger['name'] = used_name

        logger = WandbLogger(**args_logger)

        logger.watch(
            model,
            log='gradients',
            log_freq=10
        )

        for f in glob.glob('lightning_*.py'):
            wandb.save(f)

        for f in glob.glob('toolbox*.py'):
            wandb.save(f)

        args.version_logger = logger.version

    else:
        logger = False
        args.version_logger = int(time.time()/1e4)

    #############
    # Callbacks #
    #############

    list_callbacks = []

    assert args.stopping_criteria in ['AP/ESeval', 'loss/ESeval']
    if args.stopping_criteria in ['AP/ESeval']:
        mode = 'max'
    else:
        mode = 'min'

    # Create checkpoint callback to save the best model
    if not args.no_checkpoint:
        path_checkpoint = os.path.join(args.modelsCheckpoints_dir, str(args.version_logger))
        print(f"\nModel will be checkpointed there: {path_checkpoint}\n")
        list_callbacks.append(ModelCheckpoint(
            dirpath=path_checkpoint,
            filename='{epoch}',
            monitor=args.stopping_criteria,
            save_top_k=1,
            mode=mode,
        ))

    # else:
    #     checkpoint_callback = False

    # Create callback EarlyStopping
    list_callbacks.append(EarlyStopping(
        monitor=args.stopping_criteria,
        min_delta=0.0,
        patience=4,
        verbose=True,
        mode=mode,
    ))

    # GPU stats monitor
    if not args.no_logger:
        list_callbacks.append(GPUStatsMonitor())

        # My logging callbacks
        list_callbacks.append(Callback_LogRuntimePerEpoch())

    ##################
    # Create trainer #
    ##################

    trainer = Trainer.from_argparse_args(
        args,
        min_epochs=4,
        logger=logger,
        callbacks=list_callbacks,
        deterministic=True,
        precision=args.myAmp,
        weights_summary='full',  # ["full","top",None]
        progress_bar_refresh_rate=50,  # TODO: fix progress bar for printing in terminal (Maybe using 0)
    )

    ##############
    # And train! #
    ##############

    trainer.fit(model, dm)

    ######################
    # Validation metrics #
    ######################

    # TODO: when PL 1.2 will be released, use trainer.validate instead (actually not sure I want to do that)
    if (not args.no_checkpoint)&(not args.fast_dev_run):
        trainer.model.suffix_testMetrics = "/validation"
        trainer.test(ckpt_path='best')


if __name__ == '__main__':

    cfg = load_cfg(path2dir='../..')
    logVersions = load_LogVersions('../..')

    #################
    # CREATE PARSER #
    #################

    parser = ArgumentParser()

    # add program level args
    parser.add_argument('--torch_seed', type=int, default=42)

    parser.add_argument('--model_family', type=str, default='Siamese') # [Siamese, OmicsRegression]

    parser.add_argument('--filtering_goldStandard', type=str, default='all')
    parser.add_argument('--removeHubs_goldStandard', type=str,
                        default='keepAll')  # 'keepAll' or str(int) matching the files in the GS dict
    parser.add_argument('--which_goldStandard', type=str, default='main') # ['main', 'otherGoldStandard_controlledOverlap', 'otherGoldStandard_controlledOverlap-hub-hub'] etc.
    parser.add_argument('--species', type=str, default='human')

    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, default='')  # separated by ","
    parser.add_argument('--no_logger', action='store_true')

    parser.add_argument('--modelsCheckpoints_dir', type=str, default=cfg['modelsCheckpoints'])
    parser.add_argument('--no_checkpoint', action='store_true')

    parser.add_argument('--stopping_criteria', type=str, default='AP/ESeval')
    parser.add_argument('--myAmp', type=int, default=0)

    parser.add_argument('--write_out_testPreds', action='store_true')

    parser.add_argument('--cpus_per_gpu', type=int, default=32)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    # add model specific args
    parser = PPIDataModule.add_data_specific_args(parser)
    parser = PPIDataModule_Xval.add_data_specific_args(parser)
    parser = PPIDataModule_trainNtest.add_data_specific_args(parser)
    parser = PPI_lightning.add_model_specific_args(parser)
    parser = SiameseHybrid_lightning.add_model_specific_args(parser)
    parser = OmicsRegression_lightning.add_model_specific_args(parser)

    ##################
    # PULL ARGUMENTS #
    ##################

    args = parser.parse_args()

    #######################
    # ADD FIXED ARGUMENTS #
    #######################

    args.version_goldStandard = logVersions['goldStandard']
    args.version_longVector = logVersions['featuresEngineering']['longVectors']['overall']
    args.logVersions = str(logVersions)
    args.wandb_project = 'ppi-prediction'
    args.wandbLogs_dir = cfg['TrainingLogs']
    args.XVal_outputDir = cfg['XValOutputs']

    if args.model_family == 'Siamese':
        if args.include_similarityMeasures:
            args.inputVariables = 'longVectors & SM'
            real_hybridType = 'hybrid'
        else:
            args.inputVariables = 'longVectors'
            real_hybridType = args.hybridType
        args.predictor = 'siameseNetwork_' + real_hybridType
    elif args.model_family == 'OmicsRegression':
        args.predictor = 'OmicsRegression'
        if args.include_similarityMeasures:
            args.inputVariables = 'similarityMeasures'
        else:
            args.inputVariables = 'longVectors'

    if isinstance(args.gpus, int):
        args.num_workers = max(1, args.gpus * args.cpus_per_gpu)
    else:
        args.num_workers = 1

    #############
    # RUN MODEL #
    #############

    main(args)
