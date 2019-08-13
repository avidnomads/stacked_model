"""
    Classes:
    
    Data        -- handle dataset loading, splitting, etc
    Scores      -- computing and displaying model performance metrics
    ModelTester -- utility for easily testing model parameters
    Stacked     -- stacked model class: fit a meta model to predictions of base models
    
    Other:
    
    featureImportances -- returns a pandas Series of ordered feature importances
"""


import pickle
import numpy as np
import pandas as pd


class Data:
    """ Handle dataset loading, splitting """
    
    def __init__(self, trainFilename, testFilename, otherFilename=''):
        """ Arguments:
            
            trainFilename, testFilename -- .csv files of train/test data
                which are ready for fitting
            otherFilename -- .pkl file containing dict of other relevant
                objects (e.g. a scaler, a list of feature names, etc)
        """
        
        self.trainFilename = trainFilename
        self.testFilename = testFilename
        self.otherFilename = otherFilename
        self.validateFilenames_()
        
    
    def validateFilenames_(self):
        """ Validate filenames passed to __init__ """
        
        def filetype(f): return f.split('.')[-1]
        
        if filetype(self.trainFilename) != 'csv':
            raise ValueError('trainFilename must be a .csv file')
        if filetype(self.testFilename) != 'csv':
            raise ValueError('testFilename must be a .csv file')
        if self.otherFilename and filetype(self.otherFilename) != 'pkl':
            raise ValueError('trainFilename must be a .csv file')

    def load(self):
        """ Load the prepared datasets and other dict
        
            Return X_train, y_train, X_test, y_test, other
        """
        
        self.train = pd.read_csv(self.trainFilename)
        self.test = pd.read_csv(self.testFilename)
        self.features = [c for c in self.train if c != 'target']
        if self.otherFilename:
            self.other = pickle.load(open(self.otherFilename, 'rb'))
        return (
            self.train[self.features],
            self.train['target'],
            self.test[self.features],
            self.test['target'],
            self.other,
        )
        
    @property
    def X_train(self):
        return self.train[self.features]
    @property
    def y_train(self):
        return self.train['target']
    @property
    def X_test(self):
        return self.test[self.features]
    @property
    def y_test(self):
        return self.test['target']
        
    
    def inspectData(self):
        """ Print train/test distributions """
        
        print('--- Data info ---')
        print('Train size: {} ({:.2%})'.format(
            len(self.train), len(self.train)/(len(self.train) + len(self.test))
        ))
        print('Train positive distribution: {:.2%}'.format(
            self.train['target'].sum() / len(self.train)
        ))
        print('Test size: {} ({:.2%})'.format(
            len(self.test), len(self.test)/(len(self.train) + len(self.test))
        ))
        print('Test positive distribution: {:.2%}'.format(
            self.test['target'].sum() / len(self.test)
        ))
        print('\n')
        
        
class Scores:
    """ Compute and print model performance metrics """
    
    def __init__(self, predicted, actual):
        pva = pd.DataFrame({'pred': predicted, 'actual': actual})
        self.confusion = {
            'tp': pva.loc[(pva.pred == 1) & (pva.actual == 1)],
            'fp': pva.loc[(pva.pred == 1) & (pva.actual == 0)],
            'tn': pva.loc[(pva.pred == 0) & (pva.actual == 0)],
            'fn': pva.loc[(pva.pred == 0) & (pva.actual == 1)],
        }
        for k, v in self.confusion.items():
            self.confusion[k] = len(v)
        
        self.scores = {
            'Accuracy'  : (predicted == actual).sum()/len(predicted)
        }
        try:
            self.scores['Precision'] = (
                self.confusion['tp'] / (self.confusion['tp'] + self.confusion['fp'])
            )
        except ZeroDivisionError:
            self.scores['Precision'] = np.nan
        try:
            self.scores['Recall'] = (
                self.confusion['tp'] / (self.confusion['tp'] + self.confusion['fn'])
            )
        except ZeroDivisionError:
            self.scores['Recall'] = np.nan
        
    
    def items(self):
        return self.scores.items()
    
    
    def __str__(self):
        return (
            '\n'.join(
                  ['--- Scores ---']
                + ['{}: {:.8f}'.format(k,v) for k,v in self.scores.items()]
            )
            + '\n'
        )
        
        
class ModelTester:
    """ Assess model performance """
    
    
    def __init__(self, data):
        """ data = Data object, with loaded data available """
        
        self.data = data
    
    
    def testScores(self, clf, verbose=True, prefit=True):
        """ Print test performance metrics and return model, scores """
        
        if not prefit:
            clf.fit(self.data.X_train, self.data.y_train)
        scores = Scores(
            predicted = clf.predict(self.data.X_test),
            actual = self.data.y_test
        )
        if verbose:
            print(scores)
        return clf, scores
        
    
    def testModel(self, modelClass=None, **params):
        """ Fit model with params and print test performance """
        
        print('Testing model {} with params:'.format(modelClass))
        for param, val in params.items():
            print('  {} = {}'.format(param, val))
        clf = modelClass(**params)
        return self.testScores(clf, prefit=False)
        
        
class Stacked:
    """ Train a meta-model on features + base model predictions/probas """
    
    def __init__(self, data, baseModels=None, metaModel=None):
        """ Arguments:
            
            data       -- a Data object
            baseModels -- a dict or list of models to use as base predictors
            metaModel  -- the meta model to fit to base model predictions/probas
        """
        
        self.data = data
        self.mt = ModelTester(self.data)
        if baseModels is not None:
            self.setBaseModels(baseModels)
        if metaModel is not None:
            self.metaModel = metaModel
    
    
    def setBaseModels(self, baseModels):
        """ Verify baseModels is a dict or a list, and store """
        
        try:
            # throws AttributeError if no items method
            baseModels.items()
            self.baseModels = baseModels 
        except AttributeError:
            try:
                # throws AttributeError if no enumerate method
                self.baseModels = {
                    str(i): model 
                    for i, model in enumerate(baseModels)
                }
            except AttributeError:
                raise TypeError('baseModels must have an items or enumerate method')
    
    
    def fitBaseModels(self):
        for key, model in self.baseModels.items():
            model.fit(self.data.X_train, self.data.y_train)
    
    
    def predictBase(self):
        """ Store predictions/probas of base models 
            
            Note: required to fit meta model
        """
        
        X_train = self.data.X_train
        X_test = self.data.X_test
        self.trainPredictions = pd.DataFrame(index = X_train.index)
        self.testPredictions = pd.DataFrame(index = X_test.index)
        for key, model in self.baseModels.items():
            predCol = key + '__pred'
            probCol = key + '__prob'
            self.trainPredictions[predCol] = model.predict(X_train)
            self.trainPredictions[probCol] = model.predict_proba(X_train)[:,0]
            self.testPredictions[predCol] = model.predict(X_test)
            self.testPredictions[probCol] = model.predict_proba(X_test)[:,0]
        
    
    @property
    def X_train_plus_predictions(self):
        return pd.concat((self.data.X_train, self.trainPredictions), axis=1)
    @property
    def X_test_plus_predictions(self):
        return pd.concat((self.data.X_test, self.testPredictions), axis=1)
    @property
    def features(self):
        try:
            return self.stored_features_
        except AttributeError:
            self.stored_features_ = list(self.X_test_plus_predictions.columns)
            return self.stored_features_
    
    
    def fitMetaModel(self):
        self.metaModel.fit(self.X_train_plus_predictions, self.data.y_train)
        
        
    def fitAllModels(self):
        self.fitBaseModels()
        self.predictBase()
        self.fitMetaModel()
        
        
    def predictMeta(self, train_or_test):
        """ Return meta model predictions for train and/or test set """
        
        if train_or_test == 'train':
            return self.metaModel.predict(self.X_train_plus_predictions)
        elif train_or_test == 'test':
            return self.metaModel.predict(self.X_test_plus_predictions)
        else:
            raise ValueError(
                'Invalid train_or_test="{}"'.format(train_or_test)
                + '\n(must be "train" or "test")'
            )
        
        
    def scoreMetaModel(self, train=False, test=True, verbose=True):
        """ Save and/or print scores on test and/or train sets """
        
        if train:
            self.trainScores = Scores(
                predicted = self.predictMeta('train'),
                actual = self.data.y_train
            )
            if verbose:
                print('Train scores:\n' + str(self.trainScores))
        if test:
            self.testScores = Scores(
                predicted = self.predictMeta('test'),
                actual = self.data.y_test
            )
            if verbose:
                print('Test scores:\n' + str(self.testScores))
                
    
    def scoreBaseModels(self, train=False, test=True, verbose=False):
        if train:
            self.baseTrainScores = {
                key: Scores(
                    predicted = model.predict(self.data.X_train),
                    actual = self.data.y_train
                )
                for key, model in self.baseModels.items()
            }
        if test:
            self.baseTestScores = {
                key: Scores(
                    predicted = model.predict(self.data.X_test),
                    actual = self.data.y_test
                )
                for key, model in self.baseModels.items()
            }
        if verbose:
            for key in self.baseModels:
                print('---\nScores for {}:'.format(key))
                if train:                    
                    print('<Train>\n' + str(self.baseTrainScores[key]))
                if test:                    
                    print('<Test>\n' + str(self.baseTestScores[key]))
                    
            
    def scoreAllModels(self, train=False, test=True, verbose=False):
        self.scoreBaseModels(train, test, verbose)
        self.scoreMetaModel(train, test, verbose)
                    
                    
    def scoreRowDict_(self, modelKey, modelScores):
        """ Return row dict for model, to construct score DataFrame """
        
        rowDict = modelScores.scores.copy()
        rowDict['model'] = modelKey
        return rowDict
                    
        
    def scoreDataFrame(self, train_or_test='test'):
        """ Return a DataFrame of either train or test scores
            for all base models and the meta model
        """
        
        if train_or_test == 'train':
            baseModelScores = self.baseTrainScores
            metaModelScores = self.trainScores
        elif train_or_test == 'test':
            baseModelScores = self.baseTestScores
            metaModelScores = self.testScores
        else:
            raise ValueError(
                'Invalid train_or_test="{}"'.format(train_or_test)
                + '\n(must be "train" or "test")'
            )
        
        scoreRows = [
            self.scoreRowDict_(modelKey, modelScores)
            for modelKey, modelScores in baseModelScores.items()
        ]
        scoreRows.append(self.scoreRowDict_('meta', metaModelScores))
        df = pd.DataFrame(scoreRows)
        df = df[['model'] + [c for c in df if c != 'model']]
        return df
        

def featureImportances(model, features):
    """ Return Series of sorted nonzero feature importances """
    
    fi = pd.Series(data = model.feature_importances_, index = features)
    return fi.loc[fi > 0].sort_values(ascending=False)
    

        




