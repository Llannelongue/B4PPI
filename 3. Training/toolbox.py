import os
import re
import time
import datetime
import pickle
import yaml

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import urllib
from io import StringIO
from IPython.display import display
from argparse import ArgumentParser


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
    # v2.1

    # If it's a list
    if isinstance(d, list):
        print('list: len {}'.format(len(d)))
        if n > 0:
            print(d[:n])

    # If it's an ndarray
    elif isinstance(d, np.ndarray):
        print("np.array: shape {}\n".format(d.shape))
        if n > 0:
            print(d)

    # If it's a dict
    elif isinstance(d, dict):
        #         print("\n" + "\t"*n_tabs + "Dict with keys:")
        for i, (key, value) in enumerate(d.items()):
            if i >= n:
                break;

            str_key = "-- {}: {}".format(key, type(value))

            print("\n" + "\t" * n_tabs + str_key)
            glance(value, n=n, n_tabs=n_tabs + 1)

    # If it's a dataframe
    elif isinstance(d, pd.DataFrame):
        print("DataFrame: {:,} rows \t {:,} columns".format(d.shape[0], d.shape[1]))
        if n > 0:
            display(d.head(n))

    else:
        if n > 0:
            print(d)


def load_cfg(path2dir = '..'):
    with open("{}/config.yaml".format(path2dir), 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def load_LogVersions(path2dir = '..'):
    with open('{}/logVersions.yaml'.format(path2dir), 'r') as ymlfile:
        logVersions = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return logVersions


def dump_LogVersions(logVersions):
    with open('../logVersions.yaml', 'w') as ymlfile:
        yaml.dump(logVersions, ymlfile, default_flow_style=False)

def load_modelIDs():
    with open('mapping_modelD_name.yaml', 'r') as ymlfile:
        modelIDs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return modelIDs

class dotdict(dict):
    """
    dot.notation access to dictionary attribute
    from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/28463329
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False