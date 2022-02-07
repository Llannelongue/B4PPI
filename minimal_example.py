import pandas as pd
import os

## Load data

# With sequences from a pickled pandas DataFrame

goldStandard_with_featuresSeq = pd.read_pickle(
    os.path.join('data', 'benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl')
)
display(goldStandard_with_featuresSeq)


# Without sequences from a csv

goldStandard_with_features = pd.read_csv(
    os.path.join('data', 'benchmarkingGS_v1-0_similarityMeasure_v3-1.csv'),
    sep='|'
)
display(goldStandard_with_features)


# Load data without features from csv

goldStandard = pd.read_csv(
    os.path.join('data', 'benchmarkingGS_v1-0.csv'),
    sep='|'
)
display(goldStandard)