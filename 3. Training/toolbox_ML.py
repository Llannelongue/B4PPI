
# v2.0

from toolbox import *

import sklearn.model_selection
import sklearn.metrics
import wandb
import pickle as pkl
import re

import seaborn as sns

#########################
###                   ###
###     LOAD DATA     ###
###                   ###
#########################

def load_goldStandard(args, out, whichSet, verbose=True, forceKeepAll=False):

    assert out in ['IDdict' ,'enrichedDF']
    assert args.filtering_goldStandard in ['all' ,'filtered']
    # assert (args.removeHubs_goldStandard == 'keepAll')|(RepresentsInt(args.removeHubs_goldStandard))
    assert set(whichSet) <= set([args.label_train , args.label_test])

    # Load help files
    cfg = load_cfg(path2dir='../..')
    logVersions = load_LogVersions('../..')

    # assert args.which_goldStandard in ['main', 'otherGoldStandard_controlledOverlap']
    if args.species == 'human':
        if args.which_goldStandard == 'main':
            pref = f"goldStandard_v{logVersions['goldStandard']}"
        else:
            bar = re.split("[_-]", args.which_goldStandard)
            assert len(bar) in [1,2]
            if len(bar) == 1:
                pref = f"{args.which_goldStandard}_v{logVersions['otherGoldStandard'][bar[0]]}"
            else:
                assert bar[0] == 'otherGoldStandard'
                pref = f"{args.which_goldStandard}_v{logVersions['otherGoldStandard'][bar[1]]}"

        if out == 'enrichedDF':
            foo = f"_similarityMeasure_v{logVersions['featuresEngineering']['similarityMeasure']}"
        else:
            foo = ''

    else:
        pref = f"{args.which_goldStandard}_v{logVersions['otherGoldStandard'][args.which_goldStandard]}"

        if out == 'enrichedDF':
            foo = f"_similarityMeasure_v{logVersions['featuresEngineering'][args.species]['similarityMeasure']}"
        else:
            foo = ''

    # Load data
    with open(os.path.join(cfg['outputGoldStandard'],
                           f"{pref}{foo}.pkl"),
              'rb') as f:
        GS_dict = pkl.load(f)

    # Select DataFrame of interest
    if args.which_goldStandard == 'main':
        if forceKeepAll:
            GS = GS_dict[args.filtering_goldStandard]['keepAll']
        else:
            GS = GS_dict[args.filtering_goldStandard][args.removeHubs_goldStandard]
    else:
        GS = GS_dict

    # GS = pd.read_pickle(
    #     os.path.join(
    #         cfg['outputGoldStandard'],
    #         "goldStandard_{}_v{}{}.pkl".format(
    #             args.filtering_goldStandard,
    #             logVersions['goldStandard'],
    #             foo
    #         )
    #     )
    # )

    # Add interactionID
    GS['interactionID'] = GS.uniprotID_A + GS.uniprotID_B

    # Only keep training set
    GS_2 = GS.loc[GS.trainTest.isin(whichSet)].reset_index(drop=True)

    if out == 'enrichedDF':

        GS_2.drop(['uniprotID_A' ,'uniprotID_B'], axis=1, inplace=True)
        GS_2.set_index('interactionID', inplace=True)

        return GS_2, None, None

    else:
        ### Create dict labels ###
        dict_labels = pd.Series(GS_2.isInteraction.values, index=GS_2.interactionID).to_dict()
        if verbose:
            print("\n === dict_labels \n")
            glance(dict_labels)

        ### Create dict mapping ###
        dict_mappingID = dict(zip(GS_2.interactionID.values.tolist(),
                                  GS_2.loc[:, ['uniprotID_A', 'uniprotID_B']].values.tolist()))
        if verbose:
            print("\n === dict_mappingID \n")
            glance(dict_mappingID)

        return (GS_2, dict_labels, dict_mappingID)


def load_data4prediction(nameDF, out):
    '''
    The df with IDs should be in cfg['outputPredictedPPI_IDs'] and has the columns ["uniprotID_A","uniprotID_B"]
    With potentially "isInteraction" if we know the labels
    '''

    # Load help files
    cfg = load_cfg(path2dir='../..')
    # logVersions = load_LogVersions('../..')

    # Load data
    path_df = os.path.join(cfg['outputPredictedPPI_IDs'],
                           f"PPIs2predict_{nameDF}.pkl")
    print(f"\nLoading predict data from: {path_df}")
    df = pd.read_pickle(path_df)

    print(f"Predicting on {len(df):,} samples\n")

    # Add interactionID
    df['interactionID'] = df.uniprotID_A + df.uniprotID_B

    if out == 'enrichedDF':

        mapping_df = df[['uniprotID_A' ,'uniprotID_B' ,'interactionID']].copy()

        df.drop(['uniprotID_A' ,'uniprotID_B'], axis=1, inplace=True)
        df.set_index('interactionID', inplace=True)

        return df, None, mapping_df

    else:
        ### Create dict labels (if needed) ###
        if "isInteraction" in df.columns:
            dict_labels = pd.Series(df.isInteraction.values, index=df.interactionID).to_dict()

        ### Create dict mapping ###
        dict_mappingID = dict(zip(df.interactionID.values.tolist(),
                                  df.loc[:, ['uniprotID_A', 'uniprotID_B']].values.tolist()))

        return df, dict_labels, dict_mappingID


####################################
###                              ###
###     SCIKIT-LEARN HELPERS     ###
###                              ###
####################################

def trainTest_sklearn(pipe, X_train, y_train, X_test, y_test, verbose=True):
    '''
    Make predictions for binary classifier
    :param pipe:
    :param X_train:
    :param y_train:
    :param X_test:
    :return:
    '''
    if verbose:
        print('\tTraining...')
        print(f'(on {len(X_train):,} samples)')
    t1 = time.time()
    pipe.fit(X_train, y_train)
    if verbose:
        print('\t({:.3f}s)'.format(time.time( ) -t1))

    if verbose:
        print('\tPredict...')
        print(f'(on {len(X_test):,} samples)')
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_predict = pipe.predict(X_test)

    return ({
        "y_true":  y_test,
        "y_proba": y_proba,
        "y_predict": y_predict,
        "pipe": pipe
    })

######################################
###                                ###
###     X VALIDATION FUNCTIONS     ###
###                                ###
######################################


def stratifiedXvalPartitions(
        listIDs, listLabels,
        n_splits, random_state,
        IDorIndex,
        verbose=True
):

    assert len(listIDs) == len(listLabels)
    skf = sklearn.model_selection.StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=random_state)
    # splits_idx = skf.split(np.zeros(len(idsGS)), idsGS[targetVar])
    splits_idx = skf.split(np.zeros(len(listLabels)), listLabels)

    cv_partition = []

    for i, (train_index, val_index) in enumerate(splits_idx):
        assert IDorIndex in ['ID' ,'index']

        if IDorIndex == 'ID':
            cv_partition.append({
                'train': [listIDs[i] for i in train_index],
                'validation': [listIDs[i] for i in val_index]
            })

        else:
            cv_partition.append({
                'train': train_index,
                'validation': val_index
            })

        if verbose:
            foo = [listLabels[i] for i in train_index]
            bar = [listLabels[i] for i in val_index]
            print('\tFold {}:  {:,} positive (/ {:,}) in Train\t{:,} / {:,} + in Val'.format( i+1,
                sum(foo),
                len(foo),
                sum(bar),
                len(bar),
                ))

    return cv_partition


def sampleFromXval(cv_partition, train_sampleSize, val_sampleSize, verbose=True):
    cv_partition_sample = []

    for partition in cv_partition:

        cv_partition_sample.append({
            'train': np.random.choice(partition['train'],
                                      size=train_sampleSize,
                                      replace=False),
            'validation': np.random.choice(partition['validation'],
                                           size=val_sampleSize,
                                           replace=False),
        })

        if verbose:
            print(len(cv_partition_sample[-1]['train']), len(cv_partition_sample[-1]['validation']))

    return cv_partition_sample


def Xval_sklearn(cv, X, y, pipe, verbose=True):
    yConcat_real = []
    yConcat_proba = []
    yConcat_predict = []

    for i, partition in enumerate(cv):
        if verbose:
            print('- Fold ', i + 1)
        outTT = trainTest_sklearn(pipe=pipe,
                                  X_train=X.iloc[partition['train']],
                                  y_train=y.iloc[partition['train']],
                                  X_test=X.iloc[partition['validation']],
                                  y_test=y.iloc[partition['validation']],
                                  verbose=verbose
                                  )

        yConcat_real.append(outTT['y_true'])
        yConcat_proba.append(outTT['y_proba'])
        yConcat_predict.append(outTT['y_predict'])

    yConcat_real2 = np.concatenate(yConcat_real)
    yConcat_proba2 = np.concatenate(yConcat_proba)
    yConcat_predict2 = np.concatenate(yConcat_predict)

    return ({
        'y_true': yConcat_real2,
        'y_proba': yConcat_proba2,
        'y_predict': yConcat_predict2,
        'foldsResults': list(zip(yConcat_real, yConcat_proba, yConcat_predict))
    })


#######################
###                 ###
###     LOGGING     ###
###                 ###
#######################


def wandb_logging(args, prc, roc, init=True, suffix="/validation"):
    if init:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir=args.wandbLogs_dir,
            tags=args.wandb_tags,
            config=args
        )

    recall2, precision2 = sample_curve(x=prc['recall'], y=prc['precision'], sample_size=500)

    for x, y in zip(recall2, precision2):
        # FIXME: follow-up on wandb bug: https://github.com/wandb/client/issues/1507
        wandb.log(
            {f'Recall{suffix}': x, f'Precision{suffix}': y},
            # commit=False,
        )

    fpr2, tpr2 = sample_curve(x=roc['fpr'], y=roc['tpr'], sample_size=500)

    for x, y in zip(fpr2, tpr2):
        wandb.log(
            {f'FPR{suffix}': x, f'TPR{suffix}': y},
            # commit=False,
        )

    if 'f1' in prc:
        wandb.run.summary[f"F1{suffix}"] = prc['f1']
    if 'ap' in prc:
        wandb.run.summary[f"AP{suffix}"] = prc['ap']


###########################
###                     ###
###     CURVE STUFF     ###
###                     ###
###########################

def sample_curve(x, y, sample_size):
    assert len(x) == len(y)

    idx = np.linspace(0, len(x) - 1, dtype=int, num=min(len(x), sample_size))
    x2 = [x[i] for i in idx]
    y2 = [y[i] for i in idx]

    return x2, y2


#########################################
###                                   ###
###     PRECISION RECAL FUNCTIONS     ###
###                                   ###
#########################################

def plotPRCs(prcList, myList=None):
    sns.set(font_scale=1.3)

    fig = plt.figure(figsize=(14, 8))

    for algo, prc in prcList:

        if myList is None:
            goAhead = True
        elif algo in myList:
            goAhead = True
        else:
            goAhead = False

        if goAhead:

            try:
                len(prc['thresholds'])
                plt.plot(prc['recall'],
                         prc['precision'],
                         label='{} (f1= {:0.4f}, auc = {:0.2f}, ap = {:0.2f})'.format(algo,
                                                                                      prc['f1'], prc['auc'], prc['ap']),
                         lw=2)
            except:

                plt.plot(prc['recall'], prc['precision'],
                         marker='o', markersize=4, color="red", label='{} (f1= {:0.4f})'.format(algo, prc['f1']))

    noSkill = 0.5
    plt.plot([0, 1], [noSkill, noSkill],
             linestyle='--', color=(0.6, 0.6, 0.6),
             label='random guessing')

    # plt.xlim([0.45, 1.05])
    plt.ylim([0.4, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")

    plt.title("Comparison PRC curves")

    plt.tight_layout()


def precisionRecallCurve(y_test, y_predict, y_proba, fontScale=1,
                         figsize=(10, 6), doPlot=True, title=None, xlims=[0.45, 1.05]):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_proba)

    prc_auc = sklearn.metrics.auc(recall, precision)
    prc_f1 = sklearn.metrics.f1_score(y_test, y_predict)
    prc_ap = sklearn.metrics.average_precision_score(y_test, y_proba)
    #     tn, fp, fn, tp = sklearn.metrix.confusion_matrix(y_test, y_predict).ravel()
    #     confusionMatrix = {'tn':tn,
    #                        'fp':fp,
    #                        'fn':fn,
    #                        'tp':tp
    #                       }
    df_confusion = pd.crosstab(pd.Series(y_test, name='Actual'),
                               pd.Series(y_predict, name='Predicted'))
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)

    sampleIdx = list(map(int, np.linspace(0, len(thresholds) - 1, 20)))

    prtSample = pd.DataFrame({'precision': precision[sampleIdx],
                              'recall': recall[sampleIdx],
                              'thresholds': thresholds[sampleIdx]})

    sns.set(font_scale=fontScale)
    fig = plt.figure(figsize=figsize)

    plt.plot(recall,
             precision,
             label='PRC: auc = {:0.2f}, ap = {:0.2f}, f1= {:0.2f}'.format(prc_auc, prc_ap, prc_f1))

    # no skill
    noSkill = sum(y_test) / len(y_test)
    plt.plot([0, 1], [noSkill, noSkill], linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='random guessing')

    #     plt.xlim(xlims)
    #     plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Precision-Recall curve")

    plt.tight_layout()

    if doPlot:
        plt.show()

    return ({"precision": precision,
             "recall": recall,
             "thresholds": thresholds,
             "prt": prtSample,
             "auc": prc_auc,
             "ap": prc_ap,
             "f1": prc_f1,
             "confusion": df_confusion,
             "confusion_norm": df_conf_norm,
             'plt': plt,
             })


def cvPCR(output_Xval):
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(14, 6))

    for i, (labels, y_proba, y_predict) in enumerate(output_Xval['foldsResults']):
        PRCfold = precisionRecallCurve(y_test=labels,
                                       y_predict=y_predict,
                                       y_proba=y_proba,
                                       doPlot=False)

        plt.plot(PRCfold['recall'],
                 PRCfold['precision'],
                 label='PRC fold {} (auc = {:0.2f}, ap = {:0.2f}, f1= {:0.2f})'.format(i + 1,
                                                                                       PRCfold['auc'],
                                                                                       PRCfold['ap'],
                                                                                       PRCfold['f1']))

    PRC = precisionRecallCurve(y_test=output_Xval['labels'],
                               y_predict=output_Xval['y_predict'],
                               y_proba=output_Xval['y_proba'],
                               doPlot=False)

    plt.plot(PRC['recall'],
             PRC['precision'],
             'k--',
             label='mean PRC (auc = {:0.2f}, ap = {:0.2f}, f1= {:0.2f})'.format(PRC['auc'],
                                                                                PRC['ap'],
                                                                                PRC['f1']),
             lw=2)

    noSkill = .5
    plt.plot([0, 1], [noSkill, noSkill], linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='random guessing')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.title("Precision-Recall curve")
    plt.tight_layout()
    plt.show()


# def cvPRC(X, Y, pipe, n_cv=3,
#           fontScale=1, doPlot=True, cv=None,
#           title=None, figsize=(14, 6),
#           randomStateCV = 1
#           ):

#     if cv is None:
#         cv = list(sklearn.model_selection.StratifiedKFold(n_splits=n_cv,
#                                   random_state=randomStateCV).split(X, Y))

#     if doPlot:
#         sns.set(font_scale=fontScale)
#         fig = plt.figure(figsize=figsize)

#     yConcat_test = []
#     yConcat_proba = []
#     yConcat_predict = []

#     for i, (train, test) in enumerate(cv):
#         outTT = trainTest_sklearn(pipe=pipe,
#                                     X_train=X[train],
#                                     y_train=Y[train],
#                                     X_test=X[test]
#                                     )

#         yConcat_test.append(Y[test])
#         yConcat_proba.append(outTT['y_proba'])
#         yConcat_predict.append(outTT['y_predict'])

#         PRCfold = precisionRecallCurve(y_test=Y[test],
#                                        y_predict=outTT['y_predict'],
#                                        y_proba=outTT['y_proba'],
#                                        doPlot=False)

#         if doPlot:
#             plt.plot(PRCfold['precision'],
#                      PRCfold['recall'],
#                      label='PRC fold {} (auc = {:0.2f}, ap = {:0.2f}, f1= {:0.2f})'.format(i+1,
#                                                                                            PRCfold['auc'],
#                                                                                            PRCfold['ap'],
#                                                                                            PRCfold['f1']))

#     yConcat_test = np.concatenate(yConcat_test)
#     yConcat_proba = np.concatenate(yConcat_proba)
#     yConcat_predict = np.concatenate(yConcat_predict)

#     PRC = precisionRecallCurve(y_test=yConcat_test,
#                                y_predict=yConcat_predict,
#                                y_proba=yConcat_proba,
#                                doPlot=False)

#     sampleIdx = list(map(int, np.linspace(0, len(PRC['thresholds'])-1, 20)))

#     prtSample = pd.DataFrame({'precision': PRC['precision'][sampleIdx],
#                               'recall': PRC['recall'][sampleIdx],
#                               'thresholds': PRC['thresholds'][sampleIdx]})

#     if doPlot:
#         plt.plot(PRC['precision'],
#                  PRC['recall'],
#                  'k--',
#                  label='mean PRC (auc = {:0.2f}, ap = {:0.2f}, f1= {:0.2f})'.format(PRC['auc'],
#                                                                                     PRC['ap'],
#                                                                                     PRC['f1']),
#                  lw=2)

#         # no skill
#         noSkill = sum(Y) / len(Y)
#         plt.plot([0, 1], [noSkill, noSkill], linestyle='--',
#                  color=(0.6, 0.6, 0.6),
#                  label='random guessing')

#         plt.xlim([-0.05, 1.05])
#         plt.ylim([-0.05, 1.05])
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.legend(loc="lower left")
#         if title is not None:
#             plt.title(title)
#         else:
#             plt.title("Precision-Recall curve")

#         plt.tight_layout()
#         # plt.savefig('images/06_10.png', dpi=300)
#         plt.show()

#     return({"precision": PRC['precision'],
#             "recall": PRC['recall'],
#             "thresholds": PRC['thresholds'],
#             "prt": prtSample,
#             "auc": PRC['auc'],
#             "ap": PRC['ap'],
#             "f1": PRC['f1'],
#             "confusion":PRC['confusion'],
#             "confusion_norm":PRC['confusion_norm']
#             })


def comparePRCs(X, Y, pipes, algorithms, n_cv=3, fontScale=1, title=None,
                figsize=(7, 5), xlims=[-0.05, 1.05],
                figExportPath=None
                ):
    cv = list(sklearn.model_selection.StratifiedKFold(n_splits=n_cv,
                                                      random_state=1).split(X, Y))

    sns.set(font_scale=fontScale)

    fig = plt.figure(figsize=figsize)

    for pipe, algo in zip(pipes, algorithms):
        PRC = cvPRC(X=X, Y=Y, pipe=pipe,
                    doPlot=False, cv=cv)

        plt.plot(PRC['precision'],
                 PRC['recall'],
                 label='{} (auc = {:0.3f}, ap = {:0.3f} , f1= {:0.3f})'.format(algo,
                                                                               PRC['auc'], PRC['ap'], PRC['f1']),
                 lw=2)

    # no skill
    noSkill = sum(Y) / len(Y)
    plt.plot([0, 1], [noSkill, noSkill],
             linestyle='--', color=(0.6, 0.6, 0.6),
             label='random guessing')

    plt.xlim(xlims)
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    if title is None:
        plt.title("Comparison")
    else:
        plt.title(title)

    plt.tight_layout()

    if figExportPath is not None:
        plt.savefig(figExportPath, dpi=2000)

    plt.show()


#############################
###                       ###
###     ROC FUNCTIONS     ###
###                       ###
#############################

def receiverOperatingCurve(y_test, y_predict, y_proba,
                           fontScale=1, figsize=(10, 6), doPlot=True, title=None, xlims=[0.45, 1.05]):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_proba)

    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_proba)
    roc_accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)

    sns.set(font_scale=fontScale)
    fig = plt.figure(figsize=figsize)

    plt.plot(fpr,
             tpr,
             label='ROC: auc = {:0.2f}, acc = {:0.2f}'.format(roc_auc, roc_accuracy))

    # no skill
    plt.plot([0, 1], [0, 1], linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='random guessing')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    if title is not None:
        plt.title(title)
    else:
        plt.title("ROC curve")

    plt.tight_layout()

    if doPlot:
        plt.show()

    return ({"fpr": fpr,
             "tpr": tpr,
             "thresholds": thresholds,
             "auc": roc_auc,
             "accuracy": roc_accuracy,
             'plt': plt,
             })


def plotROCs(rocList, myList=None):
    sns.set(font_scale=1.3)

    fig = plt.figure(figsize=(14, 8))

    for algo, roc in rocList:

        if myList is None:
            goAhead = True
        elif algo in myList:
            goAhead = True
        else:
            goAhead = False

        if goAhead:

            try:
                len(roc['thresholds'])
                plt.plot(roc['fpr'],
                         roc['tpr'],
                         label='{}: auc = {:0.2f}, acc = {:0.2f}'.format(algo, roc['auc'], roc['accuracy']), lw=2)
            except:

                plt.plot(roc['fpr'], roc['tpr'],
                         marker='o', markersize=4, color="red",
                         label='{} (acc = {:0.4f})'.format(algo, roc['accuracy']))

    plt.plot([0, 1], [0, 1], linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='random guessing')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")

    plt.title("Comparison ROC curves")

    plt.tight_layout()


def cvROC(output_Xval):
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(14, 6))

    for i, (labels, y_proba, y_predict) in enumerate(output_Xval['foldsResults']):
        ROCfold = receiverOperatingCurve(y_test=labels,
                                         y_predict=y_predict,
                                         y_proba=y_proba,
                                         doPlot=False)

        plt.plot(ROCfold['fpr'],
                 ROCfold['tpr'],
                 label='ROC fold {} (auc = {:0.2f}, acc = {:0.2f})'.format(i + 1,
                                                                           ROCfold['auc'],
                                                                           ROCfold['accuracy']))

    ROC = receiverOperatingCurve(y_test=output_Xval['labels'],
                                 y_predict=output_Xval['y_predict'],
                                 y_proba=output_Xval['y_proba'],
                                 doPlot=False)

    plt.plot(ROC['fpr'],
             ROC['tpr'],
             'k--',
             label='mean ROC (auc = {:0.2f}, acc = {:0.2f})'.format(ROC['auc'],
                                                                    ROC['accuracy']),
             lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='random guessing')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.show()


# def plotROC(X, Y, pipe, n_cv = 3, fontScale=1,doPlot=True, cv=None, figsize=(7,5)):

#     if cv is None:
#         cv = list(sklearn.model_selection.StratifiedKFold(n_splits=n_cv,
#                               random_state=1).split(X, Y))

#     if doPlot:
#         sns.set(font_scale=fontScale)

#         fig = plt.figure(figsize=figsize)

#     mean_tpr = 0.0
#     mean_fpr = np.linspace(0, 1, 100)
#     all_tpr = []

#     for i, (train, test) in enumerate(cv):
#         probas = pipe.fit(X[train],
#                           Y[train]).predict_proba(X[test])

#         fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y[test],
#                                          probas[:, 1],
#                                          pos_label=1)
#         mean_tpr += interp(mean_fpr, fpr, tpr)
#         mean_tpr[0] = 0.0
#         roc_auc = sklearn.metrics.auc(fpr, tpr)

#         if doPlot:
#             plt.plot(fpr,
#                      tpr,
#                      label='ROC fold %d (area = %0.2f)'
#                            % (i+1, roc_auc))

#     mean_tpr /= len(cv)
#     mean_tpr[-1] = 1.0
#     mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)

#     if doPlot:
#         plt.plot(mean_fpr, mean_tpr, 'k--',
#                  label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
#         plt.plot([0, 0, 1],
#                  [0, 1, 1],
#                  linestyle=':',
#                  color='black',
#                  label='perfect performance')
#         plt.plot([0, 1],
#                  [0, 1],
#                  linestyle='--',
#                  color=(0.6, 0.6, 0.6),
#                  label='random guessing')

#         plt.xlim([-0.05, 1.05])
#         plt.ylim([-0.05, 1.05])
#         plt.xlabel('false positive rate')
#         plt.ylabel('true positive rate')
#         plt.legend(loc="lower right")

#         plt.tight_layout()
#         # plt.savefig('images/06_10.png', dpi=300)
#         plt.show()

#     return({"fpr":mean_fpr,
#             "tpr": mean_tpr,
#             "auc": mean_auc
#            })

def compareROCs(X, Y, pipes, algorithms, n_cv=3, fontScale=1, figsize=(7, 5), exportPath=None):
    cv = list(sklearn.model_selection.StratifiedKFold(n_splits=n_cv,
                                                      random_state=1).split(X, Y))

    sns.set(font_scale=fontScale)

    fig = plt.figure(figsize=figsize)

    for pipe, algo in zip(pipes, algorithms):
        ROC = plotROC(X=X, Y=Y, pipe=pipe, doPlot=False, cv=cv)

        plt.plot(ROC['fpr'],
                 ROC['tpr'],
                 label='{} (auc = {:0.3f})'.format(algo, ROC['auc']),
                 lw=2)

    plt.plot([0, 1],
             [0, 1],
             linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='random guessing')
    plt.plot([0, 0, 1],
             [0, 1, 1],
             linestyle=':',
             color='black',
             label='perfect performance')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc="lower right")

    plt.tight_layout()

    if exportPath is not None:
        plt.savefig(exportPath)

    plt.show()


if __name__ == '__main__':
    # foo = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    # bar = [0, 0, 0, 0, 1, 1, 1]
    # baar = stratifiedXvalPartitions(
    #     listIDs=foo,
    #     listLabels=bar,
    #     n_splits=3,
    #     random_state=42,
    #     IDorIndex='ID',
    #     verbose=True
    # )
    # print(baar)

    # args = dotdict(dict(
    #     filtering_goldStandard='filtered',
    #     removeHubs_goldStandard='50'
    # ))
    # GS_2, dict_labels, dict_mappingID = load_goldStandard(
    #     args=args,
    #     out='IDdict', #['IDdict','enrichedDF']
    #     whichSet=['Xval'],
    #     verbose=True
    # )

    a, b, c = load_data4prediction(
        nameDF='test1-filtered-10_v7-0-1',
        out='IDdict'
    )

    print()



