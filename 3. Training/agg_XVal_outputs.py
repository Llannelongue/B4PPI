from toolbox import *
from toolbox_ML import wandb_logging

from argparse import ArgumentParser

import torch
import torchmetrics


def metric_curves(probas, labels, metric):
    assert metric in ['ROC', 'PRC']

    out = dict()

    if metric == 'PRC':
        precision, recall, _ = torchmetrics.functional.precision_recall_curve(probas, labels)
        out['precision'] = precision
        out['recall'] = recall

    elif metric == 'ROC':
        fpr, tpr, _ = torchmetrics.functional.roc(probas, labels)
        out['fpr'] = fpr
        out['tpr'] = tpr

    return out


def metric_value(probas, labels, metric):
    assert metric in ['F1', 'AP']

    if metric == 'AP':
        out = torchmetrics.functional.average_precision(probas, labels)

    elif metric == 'F1':
        out_bin = (probas > 0.5).float()

        out = torchmetrics.functional.f1(
            out_bin, labels.int(),
        ) # Only return the F1 score for the class 1 (because binary outcome)

    return out

def concatenate_predictions(n_folds, run_name, XVal_outputDir):
    probas = []
    labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(n_folds):
        base_fileName = f"{run_name}_fold{fold}_"

        probas.append(torch.load(os.path.join(
            XVal_outputDir,base_fileName + 'outProbas.pt'
        ),
            map_location=device
        ))
        labels.append(torch.load(os.path.join(
            XVal_outputDir,base_fileName + 'labels.pt'
        ),
            map_location=device
        ))

    # FIXME: the switch to float32 is due to a bug in PL: https://github.com/PyTorchLightning/pytorch-lightning/issues/5317
    tensor_probas = torch.cat(probas).type(torch.float32)
    tensor_labels = torch.cat(labels).type(torch.float32)

    return tensor_probas, tensor_labels


def agg_XVal(hparams):

    tensor_probas, tensor_labels = concatenate_predictions(
        n_folds=hparams.Xval_n_folds,
        run_name=hparams.run_name,
        XVal_outputDir=hparams.XVal_outputDir
    )

    prc = metric_curves(
        probas = tensor_probas,
        labels = tensor_labels,
        metric = 'PRC'
    )

    roc = metric_curves(
        probas=tensor_probas,
        labels=tensor_labels,
        metric='ROC'
    )

    prc['f1'] = metric_value(
        probas=tensor_probas,
        labels=tensor_labels,
        metric='F1'
    )

    prc['ap'] = metric_value(
        probas=tensor_probas,
        labels=tensor_labels,
        metric='AP'
    )

    if not hparams.no_logger:
        print('Logging')
        wandb_logging(
            args=hparams,
            prc=prc,
            roc=roc,
            init=True
        )

    print()


if __name__ == '__main__':
    cfg = load_cfg(path2dir='../..')

    parser = ArgumentParser()
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--no_logger', action='store_true')
    parser.add_argument('--wandb_tags', type=str, default='')
    args = parser.parse_args()

    path_hparams = os.path.join(
        cfg['XValOutputs'],
        f'{args.wandb_name}_hparams.pkl'
    )
    with open(path_hparams, 'rb') as f:
        hparams = dotdict(pickle.load(f))

    hparams.XVal_outputDir = cfg['XValOutputs']

    hparams.run_name = hparams.wandb_name
    hparams.wandb_name += '-XV'
    foo = hparams.wandb_tags.split(',') + args.wandb_tags.split(',') + ['Xval-agg']
    hparams.wandb_tags = [x for x in foo if x != '']
    hparams.no_logger = args.no_logger

    agg_XVal(hparams)
