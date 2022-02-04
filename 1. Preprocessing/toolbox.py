import re
import regex as re2
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import urllib
import requests
import re
from io import StringIO
import yaml

def createGOlist(GOcol, regex0, sep = ' '):
    # example: regex0 = r"(?<=\(GO:)[\d]+(?=\))"
    
    regex1 = re2.compile(regex0)
    
#     GO_codes = [sep.join(m) for m in (re2.findall(regex1, line) for line in GOcol)]
    GO_codes = [sep.join(map(lambda x:x.replace(' ','').replace('-','').replace('/',''), m)) for m in (re2.findall(regex1, line) for line in GOcol)]
    
    print(len(GOcol), len(GO_codes))
    
    return GO_codes

def createBoW(docs):
    vectorizer = CountVectorizer(lowercase=False, preprocessor = None, binary=True)
    BoW = vectorizer.fit_transform(docs)

    print("Shape BoW:",BoW.shape)

    return BoW, vectorizer

def glance(d, n=3):
    
    if isinstance(d,dict):
        return [v for i, v in enumerate(d.items()) if i < n]

    if isinstance(d,pd.DataFrame):
        print("{:,} rows \t {:,} columns".format(d.shape[0], d.shape[1]))
        display(d.head())
        
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
    
    # If it's a list
    if isinstance(d, list):
        print('len: {}'.format(len(d)))
        print(d[:n])
    
    # If it's an ndarray
#     elif isinstance(d, np.ndarray):
#         if len(foo.shape) == 1:
#             print(d[:n])
#         elif len(foo.shape) == 2:
#             print(d[:n,:n])
#         elif len(foo.shape) == 3:
#             print(d[:n,:n,:n])
#         else:
#             print(d)
            
    # If it's a dict
    elif isinstance(d,dict):
#         print("\n" + "\t"*n_tabs + "Dict with keys:")
        for i, (key, value) in enumerate(d.items()):
            if i>=n:
                break;
                
            str_key = "-- {}: {}".format(key, type(value))
            
            print("\n" + "\t"*n_tabs + str_key)
            glance(value, n=n, n_tabs=n_tabs+1)
                
    # If it's a dataframe
    elif isinstance(d,pd.DataFrame):
        print("DataFrame: {:,} rows \t {:,} columns".format(d.shape[0], d.shape[1]))
        display(d.head(n))
        
    else:
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