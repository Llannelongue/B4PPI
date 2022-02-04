import math

# XGB
hyperparameters_XGB = dict(
    XGB__n_estimators=dict(
        distribution='int_uniform',
        min=3,
        max=300,
    ),
    XGB__learning_rate=dict(
        distribution='log_uniform',
        min=math.log(1e-5),
        max=math.log(1)
    ),
    XGB__max_depth=dict(
        distribution='int_uniform',
        min=1,
        max=30,
    ),
    XGB__min_child_weight=dict(
        distribution='int_uniform',
        min=1,
        max=50,
    ),
    XGB__subsample=dict(
        distribution='uniform',
        min=0,
        max=1,
    ),
    XGB__colsample_bytree=dict(
        distribution='uniform',
        min=0,
        max=1,
    ),
)

#Logistic Regression
hyperparameters_LR = dict(
    LR__penalty=dict(
        values=['l1', 'l2', 'elasticnet', 'none'],
    ),
    LR__C=dict(
        distribution='log_uniform',
        min=math.log(0.001),
        max=math.log(100)
    ),
    LR__tol=dict(
        distribution='log_uniform',
        min=math.log(1e-5),
        max=math.log(1e-1)
    ),
    LR__l1_ratio=dict(
        distribution='uniform',
        min=0.,
        max=1.
    )
)

# SVM
hyperparameters_SVM = dict(
    SVM__C=dict(
        distribution='log_uniform',
        min=math.log(0.01),
        max=math.log(1000)
    ),
    SVM__kernel=dict(
        values=['linear','poly','rbf','sigmoid']
    ),
    SVM__degree=dict(
        distribution='int_uniform',
        min=2,
        max=5
    ),
    SVM__gamma=dict(
        distribution='log_uniform',
        min=math.log(1e-5),
        max=math.log(1)
    ),
)

# KNN
hyperparameters_KNN = dict(
    KNN__n_neighbors=dict(
        distribution='int_uniform',
        min=1,
        max=50
    ),
    KNN__weights=dict(
        values=['uniform','distance']
    ),
    KNN__algorithm=dict(
        values=[
            # 'ball_tree','kd_tree',
            'brute' # This is 6x slower than the other two
        ]
    ),
    KNN__leaf_size=dict(
        distribution='int_uniform',
        min=1,
        max=100
    ),
    KNN__p=dict(
        values=[1,2]
    ),
)

# Decision tree
hyperparameters_Dtree=dict(
    Dtree__criterion=dict(
        values=['gini','entropy'],
    ),
    Dtree__splitter=dict(
        values=['best','random'],
    ),
    Dtree__min_samples_split=dict(
        distribution='int_uniform',
        min=2,
        max=1000
    )
)

# Random Forest
hyperparameters_RF=dict(
    RF__n_estimators=dict(
        distribution='int_uniform',
        min=10,
        max=1000
    ),
    RF__criterion=dict(
        values=['gini','entropy']
    ),
    RF__min_samples_split=dict(
        distribution='int_uniform',
        min=2,
        max=500
    ),
    RF__max_features=dict(
        values=['sqrt','log2']
    )
)
