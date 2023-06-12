import os
__PATH = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pandas as pd

import pickle
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

class Predicter():
    """
    NOTE: Should try other models and should do something about TTC, some are < 15, many are 100000.
    """
    def __init__(self, model=None):
        self.model = MLPClassifier(solver="adam") if model is None else model
        self._fitScaler = False
        self.scaler = preprocessing.StandardScaler()
        self.numberOfFeatures = 0
        self.pickled_model = None


    def preProcess(self, x=None, y=None, targetCol="Attribute[COL]"):
        """
        Process x and y by transforming to np.array and mean scale x.
        
        Params:
            x: dataframe or list
            y: dataframe
            targetCol: str, name of column that is to be predicted
        
        Returns:
            x: np.array
            y: np.array
        """

        if isinstance(x, pd.DataFrame):

            # label encode road and scenario
            x = self.labelEncode(x)

            # only accepts numeric values in training as of now
            for c in x.columns:
                if x[c].dtype != float:
                    x = x.drop(c, axis=1)
            self.numberOfFeatures = len(x.columns)
            x.loc[x["Attribute[DTO]"]>100, "Attribute[DTO]"] = 100
            x = x.to_numpy()
            if not self._fitScaler: # Only fit the scaler once (on training data)
                print("Scaler is fitted")
                self.scaler.fit(x) # Fitting the scaler
                self._fitScaler = True
            x = self.scaler.transform(x) # Scaling the data

        elif isinstance(x, list or np.array): # Used when predicting one input at a time
            # NOTE Need to make this only accept one row: [[x,x,x,x,x,x]]
            # Also maybe check if nested and the input has the correct amount of features
            # if len(x) == self.numberOfFeatures:
            x = self.scaler.transform([x])

        if isinstance(y, pd.DataFrame):
            if targetCol and targetCol in y.columns:
                y = y[targetCol]
                y.replace(False, 0, inplace=True)
                y.replace(True, 1, inplace=True)
                y = y.to_numpy()
            else:
                print("Wrong parameters was sent in!")
        
        return x, y

    def labelEncode(self, x: pd.DataFrame, cols: list[str] = ["road", "scenario"]):
        """
        Label encoding for cols, default is road and scenario.

        Params:
            x: Dataframe
            cols: list of str, columns to label encode

        Returns:
            x: Dataframe
        """
        x[cols] = x[cols].apply(preprocessing.LabelEncoder().fit_transform)
        return x
        
        

    def fit(self, x, y):
        """
        Train the model.

        Params:
            x: np.array, preprocessed training data
            y: np.array, preprocessed training truth
        """
        self.model.fit(x, y)


    def predict(self, x):
        """
        Params:
            x: Dataframe, what to predict

        Returns:
            predictions: np.array of 0 and 1
        """
        # NOTE make it predict only one row, i.e.: [[x,x,x,x,x,x,x,x,x]]
        return self.model.predict(x)


    def getScore(self, predictions, truth):
        """"
        Shows the score of given predictions and ground truth in a confusion matrix.
        
        Prints as follows:
            True negative | False positive

            False negative | True positive

        Params:
            predictions: np.array
            truth: np.array
        
        Returns:
            cm: list[list]
        """
        _, truthProcessed = self.preProcess(y=truth)
        tot = 0
        cm = [[0, 0], [0, 0]]
        col = np.count_nonzero(truthProcessed == 1)
        
        for p, t in zip(predictions, truthProcessed):
            cm[int(t)][int(p)] += 1
            tot += 1

        prec = cm[1][1]/(cm[0][1]+cm[1][1])
        rec = cm[1][1]/(cm[1][1]+cm[1][0])
        print(f"Total: {tot}, number of collisions: {col}")
        print(f"\tTN: {cm[0][0]} \t| FP: {cm[0][1]} \n\tFN: {cm[1][0]} \t| TP: {cm[1][1]}")
        print(f"Accuracy: {round((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]), 2)}")
        print(f"Precision: {round(prec, 2)}")
        print(f"Recall: {round(rec, 2)}")
        print(f"F1: {round(2*prec*rec/(prec+rec), 2)}")
        return cm


    def saveModel(self, name, accuracy=None):
        """
        Saves the model to a file.

        Params:
            name: str, name of file
        """
        path = os.path.dirname(os.path.realpath(__file__))
        file = f"{path}/models/{name}.pkl" if not accuracy else f"{path}/models/{name}_{accuracy}.pkl"
        try:
            with open(file, "wb+") as f:
                pickle.dump(self.__dict__, f)
                print("Model saved!")
        except Exception as e:
            print("Could not save model! ->", e, file)
    
    def loadModel(self, name):
        """
        Loads a model from a file.

        Params:
            name: str, name of file
        """
        path = os.path.dirname(os.path.realpath(__file__))
        file = f"{path}/models/{name}.pkl"
        try:
            with open(file, "rb") as f:
                self.__dict__ = pickle.load(f) 
            print("Model loaded!")
        except Exception as e:
            print("Could not load model! ->", e)

