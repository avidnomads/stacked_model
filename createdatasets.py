"""
    Functions for creating and saving prepared datasets for model fitting
    
    Contains:
      createDatasetBC -- breast cancer dataset via sklearn
"""


import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

import logging
logging.basicConfig(level=logging.DEBUG)


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
