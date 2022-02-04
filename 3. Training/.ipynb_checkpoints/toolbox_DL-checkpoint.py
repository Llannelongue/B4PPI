from toolbox_ML import *

import torch
import torch.utils.data as torchData


def plotCudeUsage(device):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('\tCached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


            #################################
            ##                             ##
            ##   DATASET and DATALOADER    ##   
            ##                             ##
            #################################


def createDataDict(
        cfg, logVersions,
        hparams,
        verbose=True
):
    ## Load raw features ##

    if hparams.species == 'human':
        species_text = ''
        FE_vdict = logVersions['featuresEngineering']
    else:
        species_text = f'_{hparams.species}'
        FE_vdict = logVersions['featuresEngineering'][hparams.species]

    if hparams.normaliseBinary:
        with open(os.path.join(
                cfg['outputFeaturesEngineering'],
                f"longVectors{species_text}_imputeAll_v{FE_vdict['longVectors']['imputeAll']}.pkl"
        ), 'rb') as f:
            featuresList_norm = pickle.load(f)
    else:
        with open(os.path.join(
                cfg['outputFeaturesEngineering'],
                f"longVectors{species_text}_keepBinary_v{FE_vdict['longVectors']['keepBinary']}.pkl"
        ), 'rb') as f:
            featuresList_norm = pickle.load(f)

    if verbose:
        print("\n ### featuresList_norm.keys \n")
        print(featuresList_norm.keys())

    ## Extract uniprotID ##

    allProtID = featuresList_norm['uniprotID']
    del featuresList_norm['uniprotID']

    if verbose:
        print("\n ### after removing uniprotID \n")
        print(featuresList_norm.keys())
        print()

    ## Create dict_data ##

    dict_data = dict((key, dict()) for key in allProtID)

    for key, df in featuresList_norm.items():
        if verbose:
            print(key)

        if key not in hparams.nonNumericFeatures:
            if verbose:
                print('--- create dict of tensors')

            #       featuresTensor = torch.from_numpy(df.astype('float64'))
            featuresTensor = torch.from_numpy(df).float()

            for i, uniprotID in enumerate(allProtID):
                dict_data[uniprotID][key] = featuresTensor[i]

        else:
            if verbose:
                print('--- just CC')

            for i, uniprotID in enumerate(allProtID):
                dict_data[uniprotID][key] = df[i]

    ## Calculate dimension of concat features ##
    if verbose:
        print("\n ### Dimension of concat features \n")

    expl0 = next(iter(dict_data.values()))
    inputSizeConcat = 0

    for key in hparams.listFeatures2concat:
        if key not in hparams.nonNumericFeatures:
            feat = expl0[key]
            inputSizeConcat += feat.size()[0]
            if verbose:
                print(key, feat.size())

    return (dict_data, inputSizeConcat)

            
def embeddingChar(aa):
    listAA = ['C', 'D', 'S', 'Q', 'K', 'I', 'P', 'T', 'F', 'N', 'G', 'H', 'L', 'R', 'W', 'A', 'V', 'E', 'Y', 'M','U'] 
    
    # One hot encoding
    ans = torch.zeros(len(listAA))
    ans[listAA.index(aa)]=1
    return ans, len(listAA)


def embeddingSequence(seq):
    listEmbed = []
    
    for aa in seq:
        embchar, _ = embeddingChar(aa)
        listEmbed.append(embchar)
        
    tensorEmbed = torch.cat(listEmbed).view(len(listEmbed), -1)
    
    return(tensorEmbed)


class myDataset(torchData.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, mappingID, labels, data, listFeatures2concat, similarityMeasures):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.mappingID = mappingID
        self.data = data
        self.listFeatures2concat = listFeatures2concat

        if similarityMeasures is not None:
            self.similarityMeasures = similarityMeasures.fillna(0)
        else:
            self.similarityMeasures = similarityMeasures

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        idA, idB = self.mappingID[ID]

        # Load data and get label
        XA_dict = self.data[idA]
        XB_dict = self.data[idB]

        if self.labels is not None:
            y = self.labels[ID]
        else:
            y = None
        
        # Concat features
        XAconcat = torch.zeros(0).float()
        XBconcat = torch.zeros(0).float()
        
        for feat in self.listFeatures2concat:
            XAconcat = torch.cat((XAconcat,XA_dict[feat]),0)
            XBconcat = torch.cat((XBconcat,XB_dict[feat]),0)
            
        # sequence feature
        XAseq = embeddingSequence(XA_dict['sequence'])
        XBseq = embeddingSequence(XB_dict['sequence'])

        # similarity measures feature
        if self.similarityMeasures is not None:
            simMeasuresAB = self.similarityMeasures.loc[ID]

            if y is not None:
                assert y == simMeasuresAB['isInteraction']

            simMeasuresAB_t = torch.tensor(simMeasuresAB.loc[self.listFeatures2concat]).type_as(XAconcat)

        else:
            simMeasuresAB_t = None

        inputAB = {
            'inputA': {
                'Xconcat':XAconcat,
                'Xseq':XAseq,
                'ID':idA,
            },
            'inputB': {
                'Xconcat':XBconcat,
                'Xseq':XBseq,
                'ID': idB,
            },
            'similarityMeasuresAB':simMeasuresAB_t
        }
        return inputAB, y


class PadSequence:
    '''
    This is used to assemble all the samples from a batch
    '''
    def __call__(self, batch):
        # each element in a batch is a tuple (inputAB, y)
        
        ###################
        # Concat features #
        ###################
        
        batch_concatA = torch.stack([x[0]['inputA']['Xconcat'] for x in batch], axis=0)
        batch_concatB = torch.stack([x[0]['inputB']['Xconcat'] for x in batch], axis=0)

        #######################
        # Similarity Measures #
        #######################

        if batch[0][0]['similarityMeasuresAB'] is not None:
            batch_similarityMeasuresAB = torch.stack([x[0]['similarityMeasuresAB'] for x in batch], axis=0)
        else:
            batch_similarityMeasuresAB = None

        #####################
        # Sequence features #
        #####################

        # Get each sequence and pad it
#         sequencesA = [x[2] for x in sorted_batchA]
        sequencesA = [x[0]['inputA']['Xseq'] for x in batch]
        sequences_paddedA = torch.nn.utils.rnn.pad_sequence(sequencesA, batch_first=True, padding_value=0)
#         sequencesB = [x[3] for x in sorted_batchB]
        sequencesB = [x[0]['inputB']['Xseq'] for x in batch]
        sequences_paddedB = torch.nn.utils.rnn.pad_sequence(sequencesB, batch_first=True, padding_value=0)
        
        # Also need to store the length of each sequence
		# This is later needed in order to unpad the sequences
        lengthsA = torch.LongTensor([len(x) for x in sequencesA])
        lengthsB = torch.LongTensor([len(x) for x in sequencesB])
        
        ##########
        # Labels #
        ##########

        labels = torch.LongTensor([x[1] for x in batch])

        #######
        # IDs #
        #######

        idsA = [x[0]['inputA']['ID'] for x in batch]
        idsB = [x[0]['inputB']['ID'] for x in batch]
    
        inputAB = {
            'inputA': {
                'Xconcat':batch_concatA,
                'Xseq':sequences_paddedA,
                'Xlengths':lengthsA,
                'IDs':idsA,
            },
            'inputB': {
                'Xconcat':batch_concatB,
                'Xseq':sequences_paddedB,
                'Xlengths':lengthsB,
                'IDs': idsB,
            },
            'similarityMeasuresAB': batch_similarityMeasuresAB
        }
        
        return inputAB, labels



# def loadGoldStandard(cfg, logVersions, verbose=True):
#     idsGS = pd.read_pickle(os.path.join(cfg['outputGoldStandard'],
#                                         "goldStandardIDs_v" + logVersions['goldStandard'] + ".pkl"))
#
#     ### Create interaction IDs ###
#     idsGS['interactionID'] = idsGS.uniprotID_A + idsGS.uniprotID_B
#     if verbose:
#         print("\n === idsGS \n")
#         glance(idsGS)
#
#     ### Create dict labels ###
#     dict_labels = pd.Series(idsGS.isInteraction.values, index=idsGS.interactionID).to_dict()
#     if verbose:
#         print("\n === dict_labels \n")
#         glance(dict_labels)
#
#     ### Create dict mapping ###
#     dict_mappingID = dict(zip(idsGS.interactionID.values.tolist(),
#                               idsGS.loc[:, ['uniprotID_A','uniprotID_B']].values.tolist()))
#     if verbose:
#         print("\n === dict_mappingID \n")
#         glance(dict_mappingID)
#
#     return(idsGS, dict_labels, dict_mappingID)
        


