from MLmodels_hyperparametersSweeps import *

def return_config_ML(args, sweep=False):
    sweep_config = {
        "method": "bayes",
        # 'early_terminate':{
        #     'type':'hyperband',
        #     'max_iter':100,
        #     's':10
        # },
        'metric': {
            'name': "AP/validation",
            'goal': 'maximize'
        },
    }

    if args.wandb_name == 'LR_b01-T1':
        # Simple LogReg, no penalty
        args.predictor = 'LR'
        args.LR__penalty = 'none'
        args.LR__tol = 1e-4
        args.LR__C = 1 # not used
        args.LR__l1_ratio = None # not used

        args.which_goldStandard = 'benchmarkingGS'
        args.label_test = 'test1'

        args.wandb_tags = [
            'test2',
            # 'Xval-agg',
            'BchGS'
        ]
        args.trainNtest = True
        # args.Xval = True

    elif args.wandb_name == 'LR_b02-T1':
        # Regularised LogReg
        args.predictor = 'LR'
        args.which_goldStandard = 'benchmarkingGS'
        args.wandb_tags = ['BchGS']

        if sweep:
            args.wandb_tags += ['sweep','Xval-agg']
            args.Xval = True
            sweep_config['name'] = f"Sweep {args.wandb_name}"
            sweep_config['parameters'] = hyperparameters_LR
        else:
            args.trainNtest = True
            # args.Xval = True
            args.label_test = 'test1'
            args.wandb_tags.append('test1')

            args.LR__penalty = 'elasticnet'
            args.LR__tol = 0.0005141
            args.LR__C = 0.02395
            args.LR__l1_ratio = 0.9478

    elif args.wandb_name == 'XGB_b01-T1':
        # XGBoost
        args.predictor = 'XGB'
        args.which_goldStandard = 'benchmarkingGS'
        args.wandb_tags = ['BchGS']

        if sweep:
            args.wandb_tags += ['sweep','Xval-agg']
            args.Xval = True
            sweep_config['name'] = f"Sweep {args.wandb_name}"
            sweep_config['parameters'] = hyperparameters_XGB
        else:
            # args.Xval = True
            # args.wandb_tags.append('Xval-agg')
            args.trainNtest = True
            args.label_test = 'test1'
            args.wandb_tags.append(args.label_test)

            args.XGB__colsample_bytree = 0.8059
            args.XGB__learning_rate = 0.00002186
            args.XGB__max_depth = 29
            args.XGB__min_child_weight = 25
            args.XGB__n_estimators = 116
            args.XGB__subsample = 0.4595

    elif args.wandb_name == 'NBC_b01-T1':
        # XGBoost
        args.predictor = 'NBC'
        args.which_goldStandard = 'benchmarkingGS'
        args.wandb_tags = ['BchGS']

        if sweep:
            assert False, "There are no hyperparameters to tune in NBC."
        else:
            # args.Xval = True
            # args.wandb_tags.append('Xval-agg')
            args.trainNtest = True
            args.label_test = 'test1'
            args.wandb_tags.append(args.label_test)

    elif args.wandb_name == 'Dtree_b01-T1':
        # XGBoost
        args.predictor = 'Dtree'
        args.which_goldStandard = 'benchmarkingGS'
        args.wandb_tags = ['BchGS']

        if sweep:
            args.wandb_tags += ['sweep','Xval-agg']
            args.Xval = True
            sweep_config['name'] = f"Sweep {args.wandb_name}"
            sweep_config['parameters'] = hyperparameters_Dtree
        else:
            # args.Xval = True
            # args.wandb_tags.append('Xval-agg')
            args.trainNtest = True
            args.label_test = 'test1'
            args.wandb_tags.append(args.label_test)

            args.Dtree__criterion = 'entropy'
            args.Dtree__splitter = 'random'
            args.Dtree__min_samples_split = 895

    elif args.wandb_name == 'KNN_b01-T1':
        # XGBoost
        args.predictor = 'KNN'
        args.which_goldStandard = 'benchmarkingGS'
        args.wandb_tags = ['BchGS']

        if sweep:
            args.wandb_tags += ['sweep','Xval-agg']
            args.Xval = True
            sweep_config['name'] = f"Sweep {args.wandb_name}"
            sweep_config['parameters'] = hyperparameters_KNN
        else:
            # args.Xval = True
            # args.wandb_tags.append('Xval-agg')
            args.trainNtest = True
            args.label_test = 'test1'
            args.wandb_tags.append(args.label_test)

            args.KNN__algorithm = 'brute'
            args.KNN__leaf_size = 53
            args.KNN__n_neighbors = 35
            args.KNN__p = 2
            args.KNN__weights = 'uniform'

    elif args.wandb_name == 'RF_b01-T1':
        # XGBoost
        args.predictor = 'RF'
        args.which_goldStandard = 'benchmarkingGS'
        args.wandb_tags = ['BchGS']

        if sweep:
            args.wandb_tags += ['sweep','Xval-agg']
            args.Xval = True
            sweep_config['name'] = f"Sweep {args.wandb_name}"
            sweep_config['parameters'] = hyperparameters_RF
        else:
            # args.Xval = True
            # args.wandb_tags.append('Xval-agg')
            args.trainNtest = True
            args.label_test = 'test1'
            args.wandb_tags.append(args.label_test)

            args.RF__n_estimators = 336
            args.RF__criterion = 'gini'
            args.RF__min_samples_split = 487
            args.RF__max_features = 'log2'

    elif args.wandb_name == 'SVM_b01-T1':
        # SVM
        args.predictor = 'SVM'
        args.which_goldStandard = 'benchmarkingGS'
        args.wandb_tags = ['BchGS']

        if sweep:
            args.wandb_tags += ['sweep','Xval-agg']
            args.Xval = True
            sweep_config['name'] = f"Sweep {args.wandb_name}"
            sweep_config['parameters'] = hyperparameters_SVM
        else:
            # args.Xval = True
            # args.wandb_tags.append('Xval-agg')
            args.trainNtest = True
            args.label_test = 'test1'
            args.wandb_tags.append(args.label_test)

            args.SVM__C = 1. # default values
            args.SVM__kernel = 'rbf'
            args.SVM__degree = 3
            args.SVM__gamma = 'scale'

    elif args.wandb_name == 'GP_b01':
        # XGBoost
        args.predictor = 'GP'
        args.which_goldStandard = 'benchmarkingGS'
        args.wandb_tags = ['BchGS']

        if sweep:
            assert False, "There are no hyperparameters to tune in GP."
        else:
            args.Xval = True
            args.wandb_tags.append('Xval-agg')
            # args.trainNtest = True
            # args.label_test = 'test1'
            # args.wandb_tags.append(args.label_test)

    #### YEAST ####

    if args.wandb_name == 'LR_y-01-':
        # Simple LogReg, no penalty
        args.predictor = 'LR'
        args.LR__penalty = 'none'
        args.LR__tol = 1e-4
        args.LR__C = 1 # not used
        args.LR__l1_ratio = None # not used

        args.species = 'yeast'
        args.which_goldStandard = f'benchmarkingGS_{args.species}'

        args.label_test = 'test'

        args.wandb_tags = [
            'test',
            'yeast',
            # 'Xval-agg',
            'BchGS'
        ]
        args.trainNtest = True
        # args.Xval = True

    elif args.wandb_name == 'XGB_y-01-T':
        # XGBoost
        args.predictor = 'XGB'
        args.which_goldStandard = 'benchmarkingGS_yeast'
        args.species = 'yeast'
        args.wandb_tags = ['BchGS', 'yeast']

        if sweep:
            args.wandb_tags += ['sweep','Xval-agg']
            args.Xval = True
            sweep_config['name'] = f"Sweep {args.wandb_name}"
            sweep_config['parameters'] = hyperparameters_XGB
        else:
            # args.Xval = True
            # args.wandb_tags.append('Xval-agg')
            args.trainNtest = True
            args.label_test = 'test'
            args.wandb_tags.append(args.label_test)

            args.XGB__colsample_bytree = 0.7087
            args.XGB__learning_rate = 0.00001129
            args.XGB__max_depth = 26
            args.XGB__min_child_weight = 3
            args.XGB__n_estimators = 244
            args.XGB__subsample = 0.9318

    elif args.wandb_name == 'NBC_y-01-T':
        # XGBoost
        args.predictor = 'NBC'
        args.which_goldStandard = 'benchmarkingGS_yeast'
        args.species = 'yeast'
        args.wandb_tags = ['BchGS','yeast']

        if sweep:
            assert False, "There are no hyperparameters to tune in NBC."
        else:
            # args.Xval = True
            # args.wandb_tags.append('Xval-agg')
            args.trainNtest = True
            args.label_test = 'test'
            args.wandb_tags.append(args.label_test)


    #### OLD
    elif args.wandb_name == 'LR_10-CO-LL-T1':
        args.predictor = 'LR'
        args.LR__C = 0.001206
        args.LR__l1_ratio = 0.1836
        args.LR__penalty = 'none'
        args.LR__tol = 0.003588

        args.which_goldStandard = 'otherGoldStandard_controlledOverlap-lone-lone'
        args.filtering_goldStandard = 'filtered'
        # args.removeHubs_goldStandard = 'keepAll'

        args.wandb_tags = [
            'test1',
            'CO'
        ]
        # args.Xval = True
        args.trainNtest = True

    elif args.wandb_name == 'XGB_20-CO-LL':
        args.predictor = 'XGB'
        args.XGB__colsample_bytree = 0.7298
        args.XGB__learning_rate = 0.01079
        args.XGB__max_depth = 14
        args.XGB__min_child_weight = 26
        args.XGB__n_estimators = 104
        args.XGB__subsample = 0.5901

        args.which_goldStandard = 'otherGoldStandard_controlledOverlap-lone-lone'
        args.filtering_goldStandard = 'filtered'

        args.wandb_tags = [
            'test1',
            'CO'
        ]
        args.trainNtest = True
        args.checkpoint = True

    elif args.wandb_name == 'NBC_20-T1':
        args.predictor = 'NBC'

        args.which_goldStandard = 'otherGoldStandard_controlledOverlap-lone-lone'
        args.filtering_goldStandard = 'filtered'

        args.wandb_tags = [
            'test1',
            'CO'
        ]
        args.trainNtest = True
        args.checkpoint = True

    elif args.wandb_name == 'KNN_20-T1':
        args.predictor = 'KNN'
        args.KNN__algorithm = 'brute'
        args.KNN__leaf_size = 30
        args.KNN__n_neighbors = 23
        args.KNN__p = 1
        args.KNN__weights = 'uniform'

        args.which_goldStandard = 'otherGoldStandard_controlledOverlap-lone-lone'
        args.filtering_goldStandard = 'filtered'

        args.wandb_tags = [
            'test1',
            'CO'
        ]
        args.trainNtest = True
        args.checkpoint = True

    else:
        assert False, f"wanb_name {args.wandb_name} not recognised"

    return args, sweep_config