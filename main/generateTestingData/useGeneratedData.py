import os
import sys
PATH = os.path.abspath(__file__)
PATH = PATH[:PATH.find("main")-1]
sys.path.insert(0, PATH)

from pandas import DataFrame, Series, read_csv
import numpy as np
import random
import math
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn import preprocessing

from main.ML.Model import Predicter
from main.readLidar import ReadLidar


def makeDataUsable(df: DataFrame, pastImportance: int=4, rowsBeforeCol: int=5, removeCol: bool=True) -> DataFrame:
    """
    Make a usable Dataframe from a csv file from generated data.\\
    Creates pastImportance of columns for each prediction feature.\\
    Registers the "col" feature before an actual collision as 1, up to secBeforeCol back in time.\\
    If undersampleRatio is > 0, it will under sample the data, if it is 0, it will not.
    
    ### Params:
        * df: Dataframe, need to have the columns named "TTC", "DTO", "JERK" and "Speed"
        * pastImportance: int, needs to be > 0
        * rowsBeforeCol: int, needs to be > 0
        * removeCol: bool, True if the origianl collision row should be removed
    ### Returns:
        * Dataframe
    """
    colsToUse = ["TTC", "DTO", "JERK", "Speed", "asX", "asY", "asZ"]
    columns = ["Time"]
    columns += [f"{c}{i}" for c in colsToUse for i in range(1, pastImportance+1)]
    columns += ["COL", "toRemove"]

    rowsToRemove = []
    
    dataDict = {col: [] for col in columns}

    for i, row in df.iterrows():
        if row["Time"] < pastImportance-1:
            continue

        dataDict["Time"].append(row["Time"])
        for j, k in enumerate(range(i-pastImportance+1, i+1), start=1):
            for c in colsToUse:
                # Should be able to remove this if check with new data
                if (c == "TTC" or c == "DTO") and df.iloc[k][c] < 0:
                    dataDict[f"{c}{j}"].append(0)
                    continue
                dataDict[f"{c}{j}"].append(df.iloc[k][c])

        dataDict["COL"].append(row["COL"])
        dataDict["toRemove"].append(row["COL"])
        if row["COL"] == 1:
            rowsToRemove.append(i-1)
            available = int(row["Time"]-pastImportance+2)
            amount = int(rowsBeforeCol) if available >= rowsBeforeCol else available
            dataDict["COL"][-amount:] = [1]*amount

    df = DataFrame(dataDict)
    df = df[df["toRemove"] == 0] if removeCol else df
    df.drop("toRemove", axis=1, inplace=True)
    return df


def splitTrainTest(data: DataFrame, splitRatio: float=0.8) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splitting the data.

    Params:
        filename: str, name of file to split
        splitRatio: float 0-1, % of data to be testing

    Returns:
        trainX, trainY, testingX, testingY.
    """
    if not 0 < splitRatio < 1: raise ValueError("SplitRatio must be between 0 and 1!")

    # Shuffle data
    data = data.sample(frac=1, random_state=1)

    temp = data.copy()
    # Removing all non numeric values, might need to make string values into numbers
    # temp = temp.drop(["Execution","ScenarioID","Configuration_API_Description"], axis=1)
    split = int(np.floor(len(data)*splitRatio))
    print(f"Splitting at {split}, total rows: {len(data)}")

    trainX, testX = temp[:split], temp[split:]

    # trainY = concat([trainX.pop(feature) for feature in ["Attribute[COL]","Attribute[COLT]","Attribute[SAC]"]], axis=1)
    # testY = concat([testX.pop(feature) for feature in ["Attribute[COL]","Attribute[COLT]","Attribute[SAC]"]], axis=1)
    y_col = "COL"
    trainY = trainX.pop(y_col)
    testY = testX.pop(y_col)
    return trainX, trainY, testX, testY



class NewPredicter(Predicter):
    """
    NOTE: Should try other models and should do something about TTC, some are < 15, many are 100000.
    """
    def __init__(self, model=None):
        super().__init__()
        self.model = MLPClassifier(solver="adam", max_iter=1000) if model is None else model
        # self._fitScaler = False
        # self.scaler = preprocessing.StandardScaler()
        # self.numberOfFeatures = 0
        # self.pickled_model = None
        # print(f"amount of features: {self.numberOfFeatures}")
        

    def preProcess(self, x: DataFrame or list or np.array) -> np.array:
        """
        Process x and y by transforming to np.array and mean scale x.
        # TODO: Clean this up
        
        Params:
            x: dataframe, Series, list or np.array
        
        Returns:
            x: np.array
        """
        # print("Nye preProcess")
        if isinstance(x, DataFrame):
            # print("er dataframe")
            # label encode road and scenario
            # x = self.labelEncode(x)

            # only accepts numeric values in training as of now
            # for c in x.columns:
            #     if x[c].dtype != float:
            #         x = x.drop(c, axis=1)
            self.numberOfFeatures = len(x.columns)
            # x.loc[x["Attribute[TTC]"]==100000, "Attribute[TTC]"] = -1 # NOTE May be transformed to something else
            # x.loc[x["Attribute[DTO]"]==100000, "Attribute[DTO]"] = -1
            x = x.to_numpy()
            if not self._fitScaler: # Only fit the scaler once (on training data)
                print("Scaler is fitted")
                self.scaler.fit(x) # Fitting the scaler
                self._fitScaler = True
            x = self.scaler.transform(x) # Scaling the data
            return x

        # elif isinstance(x, list or np.ndarray or Series): # Used when predicting one input at a time
        if not self._fitScaler:
            print("The scaler is not fitted!")
            return None
        # print("Gjør om tester")
        # NOTE Need to make this only accept one row: [[x,x,x,x,x,x]]
        # Also maybe check if nested and the input has the correct amount of features
        # if len(x) == self.numberOfFeatures:
        # print(type(x))
        x = self.scaler.transform([x])
        # print(type(x))
        return x


    def predict(self, x): #, ttc=20, dto=20, jerk=0, speeds=[0,0,0,0,0,0], angular=[]):
        """
        Used to predict: [TTC, DTO, Jerk, road, scenario, speed1, speed2, speed3, speed4, speed5, speed6]

        TODO
        Make sure the values are correct.
        Make it so prediction only takes from each second, if updateInterval is lower then 1.

        Attribute[TTC]	Attribute[DTO]	Attribute[Jerk]	reward	road	strategy	scenario	speed1	speed2	speed3	speed4	speed5	speed6
        100000.000000	6.434714	5.04	ttc	road2	random	rain_night	5.547	4.660	4.401	4.228	3.986	3.746
        
        ### Params
            ...

        """
        # if len(x) != self.numberOfFeatures:
        #     print(f"Wrong amount of features!, got {len(x)}, need to have {self.numberOfFeatures}")
        #     return None
        
        # x = np.array([20, 20, 1, 2, 1,1,2,3,4,5,6])
        # if len(angular) < 6:
        #     x = [ttc, dto, jerk] + speeds
        # else:
        #     x = [dto, jerk] + speeds + np.array(angular).flatten().tolist()
        # print(f"Inne i predict: {x}")
        # xProcessed = self.preProcess(x)
        # print(f"x: {x}\t\tprocessed: {xProcessed}")
        # prediction = self.predict(xProcessed)[0]
        prediction = self.model.predict(x)
        return prediction


    def getScore(self, y, y_pred):
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
        acc, prec, rec, f1 = accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        print(f"Total: {len(y)}, Collisions: {np.count_nonzero(y)}")
        print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")
        print("Confusion matrix:")
        print(cm)
        return cm


    @classmethod
    def loadModel(cls, name):
        """
        Loads a model from a file.

        Params:
            name: str, name of file
        """
        file = f"{PATH}/main/ML/models/{name}.pkl"
        p = cls()
        try:
            with open(file, "rb") as f:
                p.__dict__ = pickle.load(f) 
            print("Model loaded!")
            return p
        except Exception as e:
            print("Could not load model! ->", e)
            return None


if __name__ == "__main__":
    data = makeDataUsable(read_csv(PATH + "/data/generatedData.csv"), 4, 4)
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    # print(data.head(10))
    trainX, trainY, testX, testY = splitTrainTest(data)
    print(f"trainX:{trainX.shape}, trainY:{trainY.shape}, testX:{testX.shape}, testY:{testY.shape}")
    # print(trainX.head())

    # p = NewPredicter.loadModel("xgb_2_582-11-16-201")
    p = NewPredicter.loadModel("MLPClassifierWithGeneratedData")
    # print(p.__dict__)

    # p = NewPredicter()
    # trainXpp = p.preProcess(trainX)
    # trainYpp = trainY.to_numpy()
    # testXpp = p.preProcess(testX)
    # testYpp = testY.to_numpy()
    # print(f"trainX: {type(trainXpp)}, trainY: {type(trainYpp)}, testX: {type(testXpp)}, testY: {type(testYpp)}")
    # print(trainXpp)
    # p.fit(trainXpp, trainYpp)

    pred = []
    print(p.numberOfFeatures)
    # TODO må preprocesse før predict
    for _, row in testX.iterrows():
        r = p.preProcess(row)
        pred.append(int(p.predict(r)))
    print(pred)

    p.getScore(pred, testY)
