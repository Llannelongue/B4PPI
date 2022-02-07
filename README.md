# B4PPI
*Benchmarking Pipeline for the Prediction of Protein-Protein Interactions*

[![Generic badge](https://img.shields.io/badge/Version-v1.0-blue.svg)](https://shields.io/)
  
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/purple?icon=github)](https://github.com/Naereen/badges/)

How this benchmarking pipeline has been built, and how to use it, is detailed in our preprint [here]() (please cite it if you find this work useful!).

A minimal example is available [here](minimal_example.py), and the list of requirements [there](requirements.txt).

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

<p align="center">
  <img width="323" alt="image" src="https://user-images.githubusercontent.com/22586038/152777593-d1a98260-dc0d-4f37-b56b-449d8d57f768.png">
</p>
  
- UniProtIDs are used for both proteins A and B.
- `isInteraction` is the ground truth from the [IntAct](https://www.ebi.ac.uk/intact/home) database (1 = interacting proteins, 0 = non-interacting proteins).
- `trainTest` is the split between training set (`train`), first testing set *T1* (`test1`) and second testing set *T2* (`test2`).
- Pre-processed features are explained in the manuscript.

Training and evaluation can then be done normally. The code from the preprint is in the [Training](3.%20Training/) section.

## How to cite this work

> Lannelongue L., Inouye M., **Construction of *in silico* protein-protein interaction networks across different topologies using machine learning**, 2022, BioArxiv

## Licence

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-shield]][cc-by]

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Credits 
- The code was written in Python 3.7.
- Many libraries were used, in particular [Pandas](https://pandas.pydata.org), [Numpy](https://numpy.org), [scikit-learn](https://scikit-learn.org/stable/) and [PyTorch Lightning](https://www.pytorchlightning.ai) (full list in the code and in the [requirements](requirements.txt) file).
- Plots were drawn using [Matplotlib](https://matplotlib.org), [Seaborn](https://seaborn.pydata.org) and the [MetBrewer colour palettes](https://github.com/BlakeRMills/MetBrewer).
- Logs were saved using [Weight & Bias](https://wandb.ai).
