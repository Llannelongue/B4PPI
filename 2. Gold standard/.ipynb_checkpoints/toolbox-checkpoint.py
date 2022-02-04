import re
import regex as re2
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import urllib
import requests
import re
from io import StringIO
import yaml
import numpy as np
import random
import os

        
def mappingUniprotIDs(fromID, listIDs):
    
    # Upload the list of IDs to Uniprot
    # fromID can be found here https://www.uniprot.org/help/api_idmapping

    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': fromID,
        'to': 'ACC',
        'format': 'tab',
        'query': ' '.join(listIDs)
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()

    df = pd.read_csv(StringIO(response.decode('utf-8')), sep='\t')
    return df

def glance(d, n=5, n_tabs=0):
    # v2.2
    
    # If it's a list
    if isinstance(d, list):
        print(f'list: len {len(d)}')
        if n>0:
            print(d[:n])
    
    # If it's an ndarray
    elif isinstance(d, np.ndarray):
        print(f"np.array: shape {d.shape}\n")
        if n>0:
            print(d)
            
    # If it's a dict
    elif isinstance(d,dict):
        print(f"Dict: {len(d.keys()):,} keys")
        for i, (key, value) in enumerate(d.items()):
            if i>=n:
                break;
                
            str_key = f"-- {key}: {type(value)}"
            
            print("\n" + "\t"*n_tabs + str_key)
            glance(value, n=n, n_tabs=n_tabs+1)
                
    # If it's a dataframe
    elif isinstance(d,pd.DataFrame):
        print(f"DataFrame: {d.shape[0]:,} rows \t {d.shape[1]:,} columns")
        if n>0:
            display(d.head(n))
        
    else:
        if n>0:
            print(d)
            
def load_cfg():
    with open("../config.yaml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg

def load_LogVersions():
    with open('../logVersions.yaml', 'r') as ymlfile:
        logVersions = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return logVersions

def dump_LogVersions(logVersions):
    with open('../logVersions.yaml', 'w') as ymlfile:
        yaml.dump(logVersions, ymlfile, default_flow_style=False)
        
class dotdict(dict):
    """
    dot.notation access to dictionary attribute
    from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/28463329
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def sampleNIPs(IDs2sample, seed, targetSampleSize, referenceIntAct, otherRefDf=None):
    """
    If IDs2sample is a tuple of 2 lists, samples one element from each
    """
    # v2.0 (16/11/2021)
    sampleSize = int(targetSampleSize*1.5)
    
    interactors = []
    
    random.seed(seed)
    
    for i in range(sampleSize):
        
        if isinstance(IDs2sample[0], list):
            interactors.append([
                random.choice(IDs2sample[0]),
                random.choice(IDs2sample[1])
            ])
        else:
            interactors.append(random.sample(IDs2sample, 2))
    
    # Order each pair alphabetically to be consistent
    for i in range(len(interactors)):
        interactors[i].sort()
        
    # Create df
    negGS = pd.DataFrame.from_records(interactors, columns = ["uniprotID_A", "uniprotID_B"])
    
    # Remove self interactions
    negGS_0 = negGS.loc[negGS.uniprotID_A != negGS.uniprotID_B].copy()
    print("Removed {:,} self-interactions".format(len(negGS)-len(negGS_0)))
    
    # drop duplicates
    negGS_1 = negGS_0.drop_duplicates()
    print("Removed {:,} duplicates".format(len(negGS_0)-len(negGS_1)))
    
    # compare to IntAct to exclude known false negatives
    negGS_2 = negGS_1.merge(referenceIntAct, how = "left", on = ['uniprotID_A', 'uniprotID_B'])
    negGS_3 = negGS_2.loc[negGS_2['intact-miscore'].isnull()].copy()
    negGS_3.drop(['intact-miscore'], axis=1, inplace=True)
    print("Removed {:,} known false negatives".format(len(negGS_2)-len(negGS_3)))
    
    # compare to other datasets
    if otherRefDf is not None:
        memo = len(negGS_3)
        foo = otherRefDf.copy()
        foo['isThere'] = 1
        
        bar = negGS_3.merge(foo, how = "left", on = ['uniprotID_A', 'uniprotID_B'])
        negGS_3 = bar.loc[bar['isThere'].isnull()].copy()
        negGS_3.drop(['isThere'], axis=1, inplace=True)
        
        print("Removed {:,} based on other ref dataset".format(memo-len(negGS_3)))
        
    
    # keep only the target number of interactions
    assert len(negGS_3) >= targetSampleSize
    negGS_4 = negGS_3.iloc[:targetSampleSize]
    
    print("Final number of interactions: {:,}".format(len(negGS_4)))
    
    ## sanity checks
    # right number of interactors
    assert len(interactors) == sampleSize
    # no self interactions
    foo = negGS_0.loc[negGS_0.uniprotID_A == negGS_0.uniprotID_B]
    assert len(foo) == 0
    # no duplicates
    assert ~negGS_1.duplicated().any()
    # no false positives
    foo = negGS_3.merge(referenceIntAct, how = "inner", on = ['uniprotID_A', 'uniprotID_B'])
    assert len(foo) == 0
    if otherRefDf is not None:
        foo = negGS_3.merge(otherRefDf, how = "inner", on = ['uniprotID_A', 'uniprotID_B'])
        assert len(foo) == 0
    # enough interactions
    assert len(negGS_4) == targetSampleSize
    
    return negGS_4




def hubStatus(df, out=False):
    outDict = dict()
    
    print("Hub status:")
    
    foo = df.loc[df.hubType == 2]
    a = len(foo)
    b = a/len(df)
    print(f"- {a:,} hub-hub interactions ({b:.2%} - {foo.isInteraction.sum()/a:.2%} positive)")
    outDict['hub-hub'] = (a,b)
    
    foo = df.loc[df.hubType == 1]
    a = len(foo)
    b = a/len(df)
    print(f"- {a:,} hub-lone interactions ({b:.2%} - {foo.isInteraction.sum()/a:.2%} positive)")
    outDict['hub-lone'] = (a,b)
    
    foo = df.loc[df.hubType == 0]
    a = len(foo)
    b = a/len(df)
    print(f"- {a:,} lone-lone interactions ({b:.2%} - {foo.isInteraction.sum()/a:.2%} positive)")
    outDict['lone-lone'] = (a,b)
    
    if out:
        return outDict

def train_overlap(df, out=False):
    outDict = dict()
        
    print("Overlap with train:")
    
    foo = df.loc[df.isInTrain == 2]
    a = len(foo)
    b = a/len(df)
    print(f"- {a:,} total overlap with train, both proteins already seen ({b:.2%} - {foo.isInteraction.sum()/a:.2%} positive)")
    outDict['total overlap'] = (a,b)
    
    foo = df.loc[df.isInTrain == 1]
    a = len(foo)
    b = a/len(df)
    print(f"- {a:,} partial overal, only one protein seen ({b:.2%} - {foo.isInteraction.sum()/a:.2%} positive)")
    outDict['partial overlap'] = (a,b)
    
    foo = df.loc[df.isInTrain == 0]
    a = len(foo)
    b = a/len(df)
    print(f"- {a:,} no overlap ({b:.2%} - {foo.isInteraction.sum()/a:.2%} positive)")
    outDict['no overlap'] = (a,b)
    
    if out:
        return outDict

def find_overlapStatus(df, trainSet):
    if trainSet is not None:
        ids_train = list(set(pd.concat([trainSet.uniprotID_A,trainSet.uniprotID_B])))
    else:
        ids_train = []
        
    AisInTrain = df.uniprotID_A.isin(ids_train).astype(int)
    BisInTrain = df.uniprotID_B.isin(ids_train).astype(int)
    isInTrain = AisInTrain + BisInTrain
    
    return AisInTrain, BisInTrain, isInTrain

def EDA_predSet(df0, list_hubs, trainSet, PosNeg=True, Hub=True, Overlap=True, out=False):
    
    outDict = dict()
    
    df = df0.copy()
    
    print(f"{len(df):,} interactions ({df.isInteraction.sum()/len(df):.2%} positive)")
    print()
    
    df['AisHub'] = df.uniprotID_A.isin(list_hubs).astype(int)
    df['BisHub'] = df.uniprotID_B.isin(list_hubs).astype(int)
    df['hubType'] = df.AisHub + df.BisHub
    
#     if trainSet is not None:
#         ids_train = set(pd.concat([trainSet.uniprotID_A,trainSet.uniprotID_B]))
#     else:
#         ids_train = []
        
#     df['AisInTrain'] = df.uniprotID_A.isin(ids_train).astype(int)
#     df['BisInTrain'] = df.uniprotID_B.isin(ids_train).astype(int)
#     df['isInTrain'] = df.AisInTrain + df.BisInTrain
    
    df['AisInTrain'], df['BisInTrain'], df['isInTrain'] = find_overlapStatus(df, trainSet)
    
#     display(df)
    
    if Hub:
        foo = hubStatus(df, out=True)
        outDict.update(foo)
        print()
    if Overlap:
        foo = train_overlap(df, out=True)
        outDict.update(foo)
        print()
    
    if PosNeg:
        for x,y in zip(['Positive','Negative'],[1,0]):
            print(f"### {x} only ###\n")
            foo = df.loc[df.isInteraction == y]
            if Hub:
                hubStatus(foo)
                print()
            if Overlap:
                train_overlap(foo)
                print()
    
    if Hub&Overlap:
        for x,y in zip(['Hub-Hub','Hub-Lone','Lone-Lone'],[2,1,0]):
            print(f"### {x} only ###\n")
            foo = df.loc[df.hubType == y]
            train_overlap(foo)
            print()

        for x,y in zip(['Complete overlap','Partial overlap','No overlap'],[2,1,0]):
            print(f"### {x} only ###\n")
            foo = df.loc[df.isInTrain == y]
            hubStatus(foo)
            print()

    if out:
        return outDict
            
def AddSimilarityMeasures(df, cfg, logVersions, df_path=None):
    
    if df_path is None:
        path2use = f"similarityMeasures_v{logVersions['featuresEngineering']['similarityMeasure']}.pkl"
    else:
        path2use = df_path
    
    df_features = pd.read_pickle(
        os.path.join(
            cfg['outputFeaturesEngineering'],
            path2use
        )
    )
    df_features['isThere'] = 1
    assert ~df_features.duplicated(subset=["uniprotID_A","uniprotID_B"]).any()
    
    foo = df.merge(
        df_features,
        how = 'left',
        on = ["uniprotID_A","uniprotID_B"]
    )
    
    assert len(foo) == len(df)  
    assert foo.isThere.isna().sum() == 0
    assert ~foo.duplicated(subset=["uniprotID_A","uniprotID_B"]).any()
    
    foo.drop(['isThere'], axis=1, inplace=True)
    
    return foo