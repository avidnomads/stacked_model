"""
    Compare stacked model to basic models on HTRU2 pulsar dataset
"""


import pickle
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from createdatasets import createDatasetHTRU2
from stackedmodel import Data, Stacked, featureImportances


TRAIN_FILENAME = 'htru2_train_data.csv'
TEST_FILENAME = 'htru2_test_data.csv'
OTHER_FILENAME = 'htru2_other.pkl'

""" Uncomment if you need to recreate the dataset
"""
#createDatasetHTRU2(TRAIN_FILENAME, TEST_FILENAME, OTHER_FILENAME, 0.2)
"""
"""

data = Data(TRAIN_FILENAME, TEST_FILENAME, OTHER_FILENAME)
X_train, y_train, X_test, y_test, scaler = data.load()
features = data.features
data.inspectData()

baseModels = {
    'l1_regression_strong': 
        LogisticRegression(
            penalty = 'l1',
            C = 0.03,
            solver = 'liblinear',
        ),
    'l1_regression_weak': 
        LogisticRegression(
            penalty = 'l1',
            C = 0.8,
            solver = 'liblinear',
        ),
    'l2_regression_strong': 
        LogisticRegression(
            penalty = 'l2',
            C = 0.15,
            solver = 'liblinear',
        ),
    'l2_regression_weak': 
        LogisticRegression(
            penalty = 'l2',
            C = 2,
            solver = 'liblinear',
        ),
}

for depth in (4, 8, 12, 16, 32):
    key = 'tree_depth_' + str(depth)
    baseModels[key] = DecisionTreeClassifier(
        max_depth = depth,
        min_samples_leaf = 20,
    )

for k in (1, 2, 4, 8, 16, 64, 128):
    key = str(k) + '_nn'
    baseModels[key] = KNeighborsClassifier(n_neighbors = k)

metaModel = RandomForestClassifier(
    max_depth = 16,
    min_samples_leaf = 5,
    max_features = 0.75,
    n_estimators = 250,
    random_state = 314159,
)

stackedModel = Stacked(data, baseModels, metaModel=metaModel)

print('Fitting models...')
stackedModel.fitAllModels()
print('Scoring...')
stackedModel.scoreAllModels(train=False, test=True, verbose=False)
print('  Done')

print('\nMeta model feature importances (top 20):')
print(featureImportances(stackedModel.metaModel, stackedModel.features)[:20])

print('\nResults:')
print(stackedModel.scoreDataFrame())


def tryMetaModel(model):
    """ Try a different meta model """
    
    stackedModel.metaModel = model
    stackedModel.fitMetaModel()
    stackedModel.scoreMetaModel(train=False, test=True, verbose=True)
    
print('\nTrying different meta models:')
print('  1. l1 regression')
tryMetaModel(LogisticRegression(penalty='l1', C=0.0009))
print('  2. l2 regression')
tryMetaModel(LogisticRegression(penalty='l2', C=0.0007))
for i, k in enumerate((1, 2, 4, 8, 16)):
    print('  {}. {}-neighbors'.format(2 + (i + 1), k))
    tryMetaModel(KNeighborsClassifier(n_neighbors=k))


