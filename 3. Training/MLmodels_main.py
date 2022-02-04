from toolbox_ML import *

import sklearn.preprocessing
import sklearn.impute
import sklearn.pipeline
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neural_network
import sklearn.neighbors
import sklearn.tree
import sklearn.gaussian_process
import sklearn.svm

import statsmodels.api as sm

from joblib import dump, load

import xgboost as xgb

# from MLmodels_hyperparametersSweeps import *
from MLmodels_configs import return_config_ML

import multiprocessing
import pathlib


class ML4PPI:
    '''
    Main class with all the function needed to train and evaluate ML models for PPI predictions.
    '''

    def __init__(self, args):
        self.args = args

    def prepare_data(self):
        '''
        Load training set and create cross-validation partitions.
        :return:
        '''

        if self.args.Xval:
            # Load training set
            self.GS, _, _ = load_goldStandard(
                args=self.args,
                out='enrichedDF',
                whichSet=[self.args.label_train],
                verbose=self.args.verbose,
            )
            print(f"\n{len(self.GS):,} observations in the Gold Standard (subset={args.label_train}).\n")
            self.GS.drop(['trainTest'], axis=1, inplace=True)

            # Cross-validation
            # print('\n--- Cross-validation splits ---')
            self.cv_partition = stratifiedXvalPartitions(
                listIDs=self.GS.index.to_list(),
                listLabels=self.GS.isInteraction.to_list(),
                IDorIndex='index',
                n_splits=self.args.Xval_n_folds,
                random_state=self.args.Xval_random_state,
                verbose=self.args.verbose,
            )

        if self.args.trainNtest:
            self.GS, _, _ = load_goldStandard(
                args=self.args,
                out='enrichedDF',
                whichSet=[self.args.label_train,self.args.label_test],
                verbose=self.args.verbose,
            )
            self.GS.reset_index(drop=True, inplace=True)

            self.cv_partition = [dict(
                train=self.GS.loc[self.GS.trainTest == args.label_train].index.tolist(),
                validation=self.GS.loc[self.GS.trainTest == args.label_test].index.tolist()
            )]
            self.GS.drop(['trainTest'], axis=1, inplace=True)

        if args.test1KeepAll:
            GS_train, _, _ = load_goldStandard(
                args=self.args,
                out='enrichedDF',
                whichSet=[self.args.label_train],
                verbose=self.args.verbose,
            )

            GS_test, _, _ = load_goldStandard(
                args=self.args,
                out='enrichedDF',
                whichSet=[self.args.label_test],
                verbose=self.args.verbose,
                forceKeepAll=True
            )

            self.GS = GS_train.append(GS_test, ignore_index=True)

            self.cv_partition = [dict(
                train=self.GS.loc[self.GS.trainTest == self.args.label_train].index.tolist(),
                validation=self.GS.loc[self.GS.trainTest == self.args.label_test].index.tolist()
            )]
            self.GS.drop(['trainTest'], axis=1, inplace=True)

        self.Xgs = self.GS.drop('isInteraction', axis=1)
        self.ygs = self.GS.isInteraction

        # store dimensions for logging
        self.size_train = len(self.cv_partition[0]['train'])
        self.size_val = len(self.cv_partition[0]['validation'])

    def choose_predictor(self):
        '''
        Select the correct pipeline for the predictor of choice.
        :return:
        '''

        assert args.predictor in ['LR', 'NBC', 'SVM', 'XGB','KNN','Dtree','GP','RF']

        if args.predictor == 'LR':
            self.pipe = sklearn.pipeline.Pipeline([
                ('input', sklearn.impute.SimpleImputer(
                    missing_values=np.nan,
                    strategy='mean',
                    # strategy='constant',
                    # fill_value=0,
                )),
                ('scl', sklearn.preprocessing.StandardScaler()),
                ('clf', sklearn.linear_model.LogisticRegression(
                    penalty=self.args.LR__penalty,
                    solver='saga',
                    tol=self.args.LR__tol,
                    C=self.args.LR__C,
                    l1_ratio=self.args.LR__l1_ratio,
                ))
            ])

        elif args.predictor == 'NBC':
            self.pipe = sklearn.pipeline.Pipeline([
                ('imput', sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('clf', sklearn.naive_bayes.GaussianNB())
            ])

        elif args.predictor == 'SVM':
            self.pipe = sklearn.pipeline.Pipeline([
                ('imput', sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('clf', sklearn.svm.SVC(
                    probability=True,
                    C=self.args.SVM__C,
                    kernel=self.args.SVM__kernel,
                    degree=self.args.SVM__degree,
                    gamma=self.args.SVM__gamma,
                ))
            ])

        elif args.predictor == 'XGB':
            self.pipe = sklearn.pipeline.Pipeline([
                ('clf', xgb.XGBClassifier(
                    n_estimators=self.args.XGB__n_estimators,
                    learning_rate=self.args.XGB__learning_rate,
                    max_depth=self.args.XGB__max_depth,
                    min_child_weight=self.args.XGB__min_child_weight,
                    subsample=self.args.XGB__subsample,
                    colsample_bytree=self.args.XGB__colsample_bytree,
                ))
            ])

        elif args.predictor == 'KNN':
            self.pipe = sklearn.pipeline.Pipeline([
                ('imput', sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('clf', sklearn.neighbors.KNeighborsClassifier(
                    n_neighbors=self.args.KNN__n_neighbors,
                    weights=self.args.KNN__weights,
                    algorithm=self.args.KNN__algorithm,
                    leaf_size=self.args.KNN__leaf_size,
                    p=self.args.KNN__p,
                ))
            ])
        elif args.predictor == 'Dtree':
            self.pipe = sklearn.pipeline.Pipeline([
                ('imput', sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('clf', sklearn.tree.DecisionTreeClassifier(
                    criterion=self.args.Dtree__criterion,
                    splitter=self.args.Dtree__splitter,
                    min_samples_split=self.args.Dtree__min_samples_split,
                ))
            ])

        elif args.predictor == 'GP':
            self.pipe = sklearn.pipeline.Pipeline([
                ('imput', sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('clf', sklearn.gaussian_process.GaussianProcessClassifier())
            ])

        elif args.predictor == 'RF':
            self.pipe = sklearn.pipeline.Pipeline([
                ('imput', sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('clf', sklearn.ensemble.RandomForestClassifier(
                    n_estimators=self.args.RF__n_estimators,
                    criterion=self.args.RF__criterion,
                    min_samples_split=self.args.RF__min_samples_split,
                    max_features=self.args.RF__max_features
                ))
            ])

        else:
            self.pipe = None

        self.args.pipe = self.pipe

    def further_analysis(self):
        '''
        Performs further analysis post training and testing, such as studying the coefficients.
        :return:
        '''

        if args.predictor == 'LR':
            coefficients = pd.DataFrame({
                'Variable':self.Xgs.columns,
                'Coefficient':self.pipe['clf'].coef_.flatten()
            })

            coefficients = coefficients.sort_values('Coefficient', ascending=False)

            print()
            print(coefficients)
            print()

            ### Calculate regression coefficients with statsmodel
            partition = self.cv_partition[0]
            X_train = self.Xgs.iloc[partition['train']]
            y_train = self.ygs.iloc[partition['train']].reset_index(drop=True)

            pipe = sklearn.pipeline.Pipeline([
                ('input', sklearn.impute.SimpleImputer(
                    missing_values=np.nan,
                    strategy='mean',
                )),
                ('scl', sklearn.preprocessing.StandardScaler()),
            ])
            X_train_2 = pipe.fit_transform(X_train)
            X_train_3 = pd.DataFrame(X_train_2, columns=X_train.columns)

            logit_mod = sm.Logit(y_train, X_train_3)
            logit_res = logit_mod.fit()

            print(logit_res.summary())

    def one_run(self, idx_partition):
        '''
        Train and validate the model on one cross-validation partition.
        :param idx_partition: [int] index of the cross-validation partition to use. Must be in [0, `Xval_n_folds`-1].
        :return:
        '''

        assert isinstance(idx_partition, int)
        assert idx_partition in range(self.args.Xval_n_folds)

        # Create working datasets
        partition = self.cv_partition[idx_partition]

        X_train = self.Xgs.iloc[partition['train']]
        y_train = self.ygs.iloc[partition['train']]
        X_val = self.Xgs.iloc[partition['validation']]
        y_val = self.ygs.iloc[partition['validation']]

        # Sanity checks
        assert len(set(X_train.index)) == len(X_train.index)
        assert len(set(X_train.index) & set(X_val.index)) == 0

        outPredictions = trainTest_sklearn(
            pipe=self.pipe,
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val
        )

        return outPredictions

    def Xval(self):
        '''
        Runs a full cross-validation.
        :return:
        '''
        outPredictions = Xval_sklearn(
            cv=self.cv_partition,
            X=self.Xgs,
            y=self.ygs,
            pipe=self.pipe,
            verbose=self.args.verbose
        )

        return outPredictions

    def logging(self, outPrediction, initWandb):
        '''
        Logs PRC, ROC and various metrics into WandB.
        :param outPrediction: [dict] predictions from `one_run` or `Xval`.
        :param initWandb: [bool] whether to initialise the WandB logger.
        :return:
        '''
        prc = precisionRecallCurve(
            y_test=outPrediction['y_true'],
            y_predict=outPrediction['y_predict'],
            y_proba=outPrediction['y_proba'],
            doPlot=False
        )

        roc = receiverOperatingCurve(
            y_test=outPrediction['y_true'],
            y_predict=outPrediction['y_predict'],
            y_proba=outPrediction['y_proba'],
            doPlot=False
        )

        wandb_logging(
            args=self.args,
            prc=prc,
            roc=roc,
            init=initWandb
        )

        wandb.run.summary['trainingSet_size'] = self.size_train
        wandb.run.summary['validationSet_size'] = self.size_val

        self.version_logger = wandb.run.id

class oneRound():
    '''
    Class used to train and evaluate a specific ML model.
    '''

    def __init__(self, args):
        '''
        Parameters of the model we want to train/test. Edited manually for each run.
        :param args:
        '''
        self.args, _ = return_config_ML(args)

    def main(self):

        ####### Roll! ###########

        print('Initialise...')
        trainer = ML4PPI(self.args)
        print('Prepare data...')
        trainer.prepare_data()
        print('Choose predictor...')
        trainer.choose_predictor()
        print(trainer.pipe)

        if self.args.Xval:
            print('Cross-validation...')
            outPredictions = trainer.Xval()
        else:
            print('One run...')
            outPredictions = trainer.one_run(0)

            trainer.further_analysis()

        if not self.args.no_logger:
            print('Logging ')
            trainer.logging(outPredictions, initWandb=True)
        else:
            trainer.version_logger = int(time.time()/1e4)

        if not self.args.Xval:
            if args.checkpoint:
                path0 = os.path.join(
                    args.modelsCheckpoints_dir,
                    str(trainer.version_logger)
                )

                p = pathlib.Path(path0)
                p.mkdir(exist_ok=True) # Create subdir if it doesn't exist yet

                path_checkpoint = os.path.join(path0, 'trainedModel.joblib')
                dump(outPredictions['pipe'],path_checkpoint)

                print(f"\nModel has been checkpointed there: {path_checkpoint}\n")

class sweepClass():

    def __init__(self, args):
        args.verbose = False
        self.args, self.sweep_config = return_config_ML(args, sweep=True)

    def main_sweep(self):
        # Initialize a new wandb run
        wandb.init(
            config=self.args,
            tags=self.args.wandb_tags,
            dir=self.args.wandbLogs_dir
        )

        args_2 = wandb.config
        ####### Roll! ###########

        trainer = ML4PPI(args_2)
        trainer.prepare_data()
        trainer.choose_predictor()

        assert args_2.Xval
        outPredictions = trainer.Xval()

        trainer.logging(outPredictions, initWandb=False)

    def sweep(self):

        sweep_id = wandb.sweep(self.sweep_config, project=self.args.wandb_project)

        # wandb bug hopefully fixed with the new version post 0.10.1 (see https://github.com/wandb/client/issues/1250)
        # wandb._secretagent(sweep_id, function=self.main_sweep) # work around wandb bug v0.10.1
        wandb.agent(sweep_id, function=self.main_sweep)

class predictClass():
    def __init__(self, args):
        self.args = args

    def prepare_predictData(self):
        df, _, self.mapping_df = load_data4prediction(
            nameDF=self.args.newData,
            out='enrichedDF',
        )

        if (self.args.initially_trained_for is not None)&(self.args.initially_trained_for != self.args.newData):
            df2 = df.copy()
            print("Adding columns to fit the inital dataset the model was trained on:")
            foo, _, _ =  load_data4prediction(
                nameDF=self.args.initially_trained_for,
                out='enrichedDF',
            )

            for foo_col in foo.columns:
                if foo_col not in df2.columns:
                    print(f"- added {foo_col}")
                    df2[foo_col] = 0

            for df_col in df2.columns:
                if df_col not in foo.columns:
                    print(f"- removed {df_col}")
                    df2.drop(df_col, axis=1, inplace=True)

            assert set(foo.columns) == set(df2.columns)

            df2 = df2[foo.columns]
        else:
            df2 = df

        self.X = df2.drop('isInteraction', axis=1)
        self.y = df2.isInteraction

    def export_predictions(self, preds):
        # Load help files
        cfg = load_cfg(path2dir='../..')

        file_out = os.path.join(
            cfg['outputPredictedPPI'],
            f"predictions__{self.args.modelID}__{self.args.newData}.pkl"
        )

        if os.path.exists(file_out):
            print("Overwritting old predictions")

        preds.to_pickle(file_out)
        print(f"Predictions exported: {file_out}")

    def predict(self):

        ### Load model from checkpoint
        if args.modelID == '':
            assert args.path2model != ''
        else:
            args.path2model = os.path.join(
                cfg['modelsCheckpoints'],
                args.modelID,
                'trainedModel.joblib'
            )

        trainedPipe = load(args.path2model)

        ### Load dataset
        self.prepare_predictData()

        ### Make predictions
        predDF = self.mapping_df.copy()

        assert (predDF.interactionID == self.X.index).all()
        assert (predDF.interactionID == self.y.index).all()

        predDF['out_probas'] = trainedPipe.predict_proba(self.X)[:, 1]
        predDF['labels'] = self.y.values

        predDF.rename(
            columns={'uniprotID_A': 'idA',
                     'uniprotID_B': 'idB'},
            inplace=True
        )
        predDF.drop(['interactionID'], axis=1, inplace=True)

        self.export_predictions(predDF)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn') # work around wandb bug v0.10.1

    # Load help files
    cfg = load_cfg(path2dir='../..')
    logVersions = load_LogVersions('../..')
    mapping_modelIDs = load_modelIDs()

    #################
    # CREATE PARSER #
    #################

    parser = ArgumentParser()

    # add program level args

    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, default='')  # separated by ","

    parser.add_argument('--filtering_goldStandard', type=str, default='filtered')
    parser.add_argument('--removeHubs_goldStandard', type=str, default='keepAll') # 'keepAll' or str(int) matching the files in the GS dict
    parser.add_argument('--which_goldStandard', type=str, default='main')
    parser.add_argument('--species', type=str, default='human')

    parser.add_argument('--checkpoint', action='store_true') # Ignored in cross-validation
    parser.add_argument('--modelsCheckpoints_dir', type=str, default=cfg['modelsCheckpoints'])

    parser.add_argument('--no_logger', action='store_true')

    parser.add_argument('--verbose', type=bool, default=True)

    # cross-validation parameters
    parser.add_argument('--Xval_n_folds', type=int, default=4)
    parser.add_argument('--Xval_random_state', type=int, default=321)
    parser.add_argument('--Xval_sample', action='store_true')

    parser.add_argument('--Xval', action='store_true')
    # parser.add_argument('--test1', action='store_true') # to remove
    parser.add_argument('--trainNtest', action='store_true')
    parser.add_argument('--test1KeepAll', action='store_true') # TODO: remove?

    ### predictors hyperparameters
    parser.add_argument('--predictor', type=str, default='')

    ### labels in GS
    parser.add_argument('--label_train', type=str, default='train')
    parser.add_argument('--label_test', type=str, default='test')

    ### Prediction

    # We only use path2model if modelID == ''
    parser.add_argument('--modelID', type=str, default='')
    parser.add_argument('--path2model', type=str, default='')
    parser.add_argument('--newData', type=str, default='')


    ##################
    # PULL ARGUMENTS #
    ##################

    args = parser.parse_args()

    args.wandb_project = 'ppi-prediction'
    args.wandbLogs_dir = cfg['TrainingLogs']
    args.modelsCheckpoints_dir = cfg['modelsCheckpoints']
    args.version_goldStandard = logVersions['goldStandard']
    args.version_similarityMeasure = logVersions['featuresEngineering']['similarityMeasure']
    args.inputVariables = 'similarityMeasures'
    args.logVersions = str(logVersions)

    #############
    # RUN MODEL #
    #############

    what2do = 'predict' # 'sweep','oneRound','predict'

    if what2do == 'sweep':
        # Sweep
        args.wandb_name = 'XGB_y-01'
        sweepClass(args).sweep()

    elif what2do == 'oneRound':
        # Normal run (just one Xval for a specific model or just train/test)
        args.wandb_name = 'NBC_y-01-T'
        args.checkpoint = True
        args.no_logger = False
        args.verbose = True
        oneRound(args).main()

    elif what2do == 'predict':
        # predict
        for modelName in [
            'LR_y-01',
            'XGB_y-01',
            # 'LR_b01',
            # 'XGB_b01',
            # 'NBC_y-01',
        ]:
            for dataset in [
                'benchmarkingGS_test1_v1-0_similarityMeasure_v3-1',
                'benchmarkingGS_test2_v1-0_similarityMeasure_v3-1',
                # 'benchmarkingGS_4hubs_test_v1-0_similarityMeasure_v3-1',

                # 'benchmarkingGS_yeast_test_v1-0_similarityMeasure_v1-0',
                # 'benchmarkingGS_yeast_4hubs_test_v1-0_similarityMeasure_v1-0'
            ]:
                # args.species = 'yeast'
                args.species = 'human'

                ID = mapping_modelIDs[modelName]
                args.modelID = ID
                # args.initially_trained_for = None
                # args.initially_trained_for = 'benchmarkingGS_test1_v1-0_similarityMeasure_v3-1'
                args.initially_trained_for = 'benchmarkingGS_yeast_test_v1-0_similarityMeasure_v1-0'

                args.newData = dataset

                predictClass(args).predict()
