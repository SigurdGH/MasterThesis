import os
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class DTPredictor():
    def __init__(self, model=MLPClassifier(solver="adam")):
        self.model = model
        self.scaler = StandardScaler()
        self._fitScaler = False
    
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
            x.loc[x["Attribute[DTO]"]==100000, "Attribute[DTO]"] = -1
            x = x.to_numpy()

            if not self._fitScaler: # Only fit the scaler once (on training data)
                self.scaler.fit(x) # Fitting the scaler
                print("Scaler is fitted")
                self._fitScaler = True
            x = self.scaler.transform(x) # Scaling the data

        elif isinstance(x, list or np.array): # Used when predicting one input at a time
            # NOTE Need to make this only accept one row: [[x,x,x,x,x,x]]
            # Also maybe check if nested and the input has the correct amount of features
            # if len(x) == self.numberOfFeatures:
            if not self._fitScaler:
                x = self.scaler.fit_transform([x])
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
        x[cols] = x[cols].apply(LabelEncoder().fit_transform)
        return x

    def fit(self, x, y):
        self.model.fit(x, y)
    
    def predict(self, x):
        return self.model.predict(x)

    def getScore(self, y, y_pred):
        acc, prec, rec, f1 = accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        print(f"Total: {len(y)}, Collisions: {np.count_nonzero(y)}")
        print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")
        print("Confusion matrix:")
        print(cm)
        return cm
    
    def saveModel(self, name, accuracy):
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