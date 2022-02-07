# B4PPI
*Benchmarking Pipeline for the Prediction of Protein-Protein Interactions*

[![Generic badge](https://img.shields.io/badge/Version-v1.0-blue.svg)](https://shields.io/)
  
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/purple?icon=github)](https://github.com/Naereen/badges/)

How this benchmarking pipeline has been built, and how to use it, is detailed in our preprint here.

A minimal example is available here, and the list of requirements there.

## How to use the gold standard

All the data files are in [`data`](data/), most of them are available as csv (`sep='|'`) and pickled pandas DataFrames (sometimes the csv file may be missing due to file size constraints on GitHub). 

The gold standard, without pre-processed features, can be loaded using:
```Python
goldStandard = pd.read_csv(
    os.path.join('data', 'benchmarkingGS_v1-0.csv'),
    sep='|'
)
```

Or with the pre-processed features:

```Python
goldStandard_with_featuresSeq = pd.read_pickle(
    os.path.join('data', 'benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl')
)
```

Training and evaluation can then be done normally. The code from the preprint is in the [Training](3.%20Training/) section.

## Licence

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-shield]][cc-by]

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Credits 
- The code was written in Python 3.7
- Many libraries were used, in particular pandas, numpy, scikit-learn and pytorch-lightning.
