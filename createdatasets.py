"""
    Functions for creating and saving prepared datasets for model fitting
    
    Contains:
      createDatasetBC -- breast cancer dataset via sklearn
      createDatasetHTRU -- pulsar dataset 
        (https://archive.ics.uci.edu/ml/datasets/HTRU2#)
"""


import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle

import logging


def createDatasetBC(trainFilename, testFilename, otherFilename, testFraction):
    """ Load the breast_cancer dataset and prepare for model fitting.
    
        Arguments:
        
        trainFilename -- .csv filename for train set
        testFilename  -- .csv filename for test set
        otherFilename -- .pkl filename for dict of containing scaler
        testFraction  -- fraction of data to preserve as test data
    """
    
    logging.debug('Loading...')
    data = load_breast_cancer()
    combined = np.hstack((data['data'], data['target'].reshape(-1,1)))
    np.random.seed(314159)
    np.random.shuffle(combined)
    df = pd.DataFrame(combined, columns=list(data.feature_names) + ['target'])
    df.target = df.target.astype(int)
    
    logging.debug('Adding features...')
    for c in df.columns:
        if 'worst' in c:
            feature = ' '.join(c.split(' ')[1:])
            worstFeature = 'worst ' + feature
            meanFeature = 'mean ' + feature
            df['deviance ' + feature] = df[meanFeature] - df[worstFeature]
            
    logging.debug('Splitting...')
    splitIndex = int((1 - testFraction)*len(df))
    train = df.iloc[:splitIndex]
    test  = df.iloc[splitIndex:]
    print('Train length: {}, test length: {}'.format(len(train), len(test)))
            
    logging.debug('Scaling...')
    features = [c for c in df.columns if c != 'target']
    scaler = StandardScaler().fit(train[features])
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    
    logging.debug('Saving...')
    train.to_csv(trainFilename, index=False)
    test.to_csv(testFilename, index=False)
    pickle.dump({'scaler': scaler}, open(otherFilename, 'wb'))
    logging.debug('  Done')
    
    
def createDatasetHTRU2(trainFilename, testFilename, otherFilename, positiveFraction):
    """ Load the HTRU2 dataset and prepare for model fitting.
    
        Arguments:
        
        trainFilename -- .csv filename for train set
        testFilename  -- .csv filename for test set
        otherFilename -- .pkl filename for dict of containing scaler
        positiveFraction -- fraction of positive class in train data
            (e.g. if .25, then train will be 25% positive calss, 75% negative class)
    """
    
    logging.debug('Loading...')
    data = pd.read_csv('HTRU_2.csv', header=None)
    data.columns = ['feature_' + str(c) for c in data.columns[:-1]] + ['target']
    
    logging.debug('Resampling train/test sets...')
    data_0 = shuffle(data.loc[data.target == 0], random_state = 314159)
    data_1 = shuffle(data.loc[data.target == 1], random_state = 314159)
    data_1_test  = data_1.iloc[:250]
    data_1_train = data_1.iloc[250:]
    class_0_train_length = int(len(data_1_train)*(1/positiveFraction - 1))
    data_0_train = data_0.iloc[:class_0_train_length]
    data_0_test  = data_0.iloc[class_0_train_length:]
    train = shuffle(pd.concat((data_0_train, data_1_train), axis=0))
    test  = shuffle(pd.concat((data_0_test, data_1_test), axis=0))
            
    logging.debug('Scaling...')
    features = [c for c in train.columns if c != 'target']
    scaler = StandardScaler().fit(train[features])
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    
    logging.debug('Saving...')
    train.to_csv(trainFilename, index=False)
    test.to_csv(testFilename, index=False)
    pickle.dump({'scaler': scaler}, open(otherFilename, 'wb'))
    logging.debug('  Done')
