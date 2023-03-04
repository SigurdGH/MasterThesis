import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

class Predicter():
    """
    NOTE: Should try other models and should do something about TTC, some are < 15, many are 100000.
    """
    def __init__(self):
        self.model = MLPClassifier(solver="adam")


    def preProcess(self, x=None, y=None, targetCol="Attribute[COL]"):
        """
        Process x and y by transforming to np.array and mean scale x.
        
        Params:
            x: dataframe
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
            x.loc[x["Attribute[TTC]"]==100000, "Attribute[TTC]"] = -1 # NOTE May be transformed to something else
            x.loc[x["Attribute[DTO]"]==100000, "Attribute[DTO]"] = -1
            # print(x.columns)
            x = x.to_numpy()
            scaler = preprocessing.StandardScaler().fit(x) # Scaling the input
            x = scaler.transform(x)

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
            cm[t][p] += 1
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

