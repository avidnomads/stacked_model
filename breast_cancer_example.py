"""
    Compare stacked model to basic models on sklearn breast_cancer dataset
"""


import pickle
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from createdatasets import createDatasetBC
from stackedmodel import Data, Stacked, featureImportances


TRAIN_FILENAME = 'bc_train_data.csv'
TEST_FILENAME = 'bc_test_data.csv'
OTHER_FILENAME = 'bc_other.pkl'

""" Uncomment if you need to recreate the dataset
"""
#createDatasetBC(TRAIN_FILENAME, TEST_FILENAME, OTHER_FILENAME, 0.3)
"""
"""

data = Data(TRAIN_FILENAME, TEST_FILENAME, OTHER_FILENAME)
X_train, y_train, X_test, y_test, scaler = data.load()
features = data.features
data.inspectData()

baseModels = {
    'l1_regression': 
        LogisticRegression(
            penalty = 'l1',
            C = 0.5,
            solver = 'liblinear',
            max_iter = 100,
        ),
    'l2_regression': 
        LogisticRegression(
            penalty = 'l2',
            C = 0.5,
            solver = 'liblinear',
            max_iter = 100,
        ),
}

for depth in (4, 12, 32):
    key = 'tree_depth_' + str(depth)
    baseModels[key] = DecisionTreeClassifier(
        max_depth = depth,
        min_samples_leaf = 5,
    )

for k in (1, 2, 4, 8, 16, 32):
    key = str(k) + '_nn'
    baseModels[key] = KNeighborsClassifier(n_neighbors = k)

metaModel = RandomForestClassifier(
    max_depth = 12,
    min_samples_leaf = 5,
    n_estimators = 200,
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

