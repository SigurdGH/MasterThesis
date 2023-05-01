import os
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class DTGenerated():
    def __init__(self, model=MLPClassifier(solver="adam")):
        self.model = model
        self.scaler = StandardScaler()
        self._fitScaler = False
        
    def preProcess(self, x=None):
        if isinstance(x, pd.DataFrame):
            # only accepts numeric values in training as of now
            for c in x.columns:
                if x[c].dtype != float:
                    x = x.drop(c, axis=1)
            x = x.to_numpy()

            if not self._fitScaler: # Only fit the scaler once (on training data)
                self.scaler.fit(x) # Fitting the scaler
                self._fitScaler = True
            x = self.scaler.transform(x) # Scaling the data

        elif isinstance(x, list or np.array):
            if not self._fitScaler:
                x = self.scaler.fit_transform([x])
            x = self.scaler.transform([x])
        


