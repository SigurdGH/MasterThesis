import pandas as pd
import numpy as np
import random

class DataManipulation():
    def __init__(self, filename: str=""):
        self.data = filename


    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, filename: str=""):
        try:
            self._data = pd.read_csv(filename)
        except:
            self._data = None
            raise FileNotFoundError(f"The file does not exist.")
    
    @data.getter
    def data(self):
        return self._data

    def ExtractAvData(self):
        # we have 6 columns av1, ..., av6, containing a list of angular velocities (x,y,z). want to split each av into 3 columns, exp. av1x, av1y, av1z, av2x, av2y ...
        av = ["av" + str(i) for i in range(1,7)]
        av_ = [f"{col}{av}" for col in av for av in ["x", "y", "z"]] # av1x, av1y, av1z, av2x ...

        angular_velocities = self._data[av].apply(lambda x: list(map(float, sum(map(lambda y: y.strip("[]").split(","), x.values), []))), axis=1)
        self._data[av_] = angular_velocities.values.tolist()
        self._data = self._data.drop(av, axis=1)
    
    def UpdateSpeedAtCollision(self):
        random.seed(1)
        res = self._data.copy()

        # for each row, get the SAC value, and the speed columns that the SAC value falls in between 
        # and update the following speed columns with a random value between 0 and the speed value
        SAC = "Attribute[SAC]"
        speeds = [f"speed{i}" for i in range(1,7)]

        d = []
        for idx, row in self._data.loc[self._data["Attribute[COL]"] == True].iterrows():
            _sac = row[SAC]
            for speed_idx, speed in enumerate(speeds):
                if speed_idx + 1 == len(speeds):
                    break
                if _sac >= row[speed] and _sac <= row[speeds[speed_idx+1]]:
                    d.append(idx)
                    # randval = 
                    # res.loc[idx, speeds[speed_idx:]] = [row[speed] + randval for _ in range(speed_idx, len(speeds))]
                    prev = res.loc[idx, speeds[speed_idx:]].values.tolist()
                    res.loc[idx, speeds] = [prev[0] + random.uniform(-1, row[speed] / 10) for _ in range(0, len(speeds)- len(prev))] + prev
                    break
        print(d)
        self._data = res

    def addFromXML(self, filename: str="") -> None:
        """
        Reads more data from XML files, as of now, only speeds at six different timstamps.

        Params:
            filename: str, name of file read from
        """
        try:
            xmlDf = pd.read_csv(filename, index_col=0)
        except:
            raise FileNotFoundError(f"The file does not exist.")
        if isinstance(self.data, pd.DataFrame):
            self._data = self.data.merge(xmlDf, how="inner", on=["ScenarioID", "road", "reward", "scenario", "strategy"], copy=False)
            self.ExtractAvData()
            self.UpdateSpeedAtCollision()
        else:
            print("Something went wrong in 'addFromXML()'!")


    def splitTrainTest(self, splitRatio: float=0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splitting the data.

        Params:
            filename: str, name of file to split
            splitRatio: float 0-1, % of data to be testing

        TODO
            Slå sammen road, scenario, strategy og reward til noe brukbart

        Returns:
            trainX, trainY, testingX, testingY.
        """
        if not 0 < splitRatio < 1: raise ValueError("SplitRatio must be between 0 and 1!")

        # Shuffle data
        self._data = self.data.sample(frac=1, random_state=1)

        temp = self._data.copy()
        # Removing all non numeric values, might need to make string values into numbers
        temp = temp.drop(["Execution","ScenarioID","Configuration_API_Description"], axis=1)
        split = int(np.floor(len(self.data)*splitRatio))
        print(f"splitting at {split}.")

        trainX, testX = temp[:split], temp[split:]

        exclude_cols = ["Attribute[COL]","Attribute[COLT]","Attribute[SAC]", "Attribute[TTC]"] # TTC is entirely correlated with COL, need to remove it.

        trainY = pd.concat([trainX.pop(feature) for feature in exclude_cols], axis=1)
        testY = pd.concat([testX.pop(feature) for feature in exclude_cols], axis=1)

        return trainX, trainY, testX, testY

    def underSample(self, sampleSize: int=1000):
        self._data = pd.concat([self._data[self._data["Attribute[COL]"] == False].sample(sampleSize, random_state=1), self._data[self._data["Attribute[COL]"] == True]])
    
    def overSample(self, sampleSize: int=1000):
        colition_df = self._data[self._data["Attribute[COL]"] == True]
        # add more samples to the collision data (duplicate)
        self._data = pd.concat([self._data, colition_df.sample(sampleSize, random_state=1, replace=True)])

    def getCompleteRow(self, index: None):
        """
        Gets the origianl row from index(es).

        Params:
            index: int or list

        Return:
            Dataframe or Series
        """
        if isinstance(index, list) or isinstance(index, int):
            return self._data.loc[index]
        return "Something went wrong."


    def getOriginalPath(self, index: int):
        """
        Gets the complete path from where the row was originally collected together with its ScenarioID.

        Params:
            index: int

        Return:
            dict{"SenarioID": str, "path": str}
        """
        if isinstance(index, int):
            row = self.getCompleteRow(index)
            return {
                "ScenarioID": row["ScenarioID"],
                "path": f"{row['strategy']}-strategy/reward-{row['reward']}/{row['road']}-{row['scenario']}-scenario-attributes.csv"
                }
        raise ValueError("Index needs to be an integer!")