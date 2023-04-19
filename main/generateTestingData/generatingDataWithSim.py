import os
import sys
from pandas import DataFrame, read_csv, concat

PATH = os.path.abspath(__file__)
PATH = PATH[:PATH[:PATH[:PATH.rfind("\\")].rfind("\\")].rfind("\\")]
sys.path.insert(0, PATH)

from main.readLidar import ReadLidar
from main.usingSim import Simulation


class GenerateData(Simulation):
    def __init__(self, map: str = "sf"):
        super().__init__(map)
    

    def spawnRandomNPCs(self, amountVehicles: int=10, amountPedestrians: int=20):
        """
        Spawns random NPCs, both pedestrians and vehicles that are roaming the map.
        
        ### Params:
            * amountVehicles: int, number of vehicles to spawn
            * amountPedestrians: int, number of pedestrians to spawn
        """
        pass


    def storeDataGenerated(self, paramsToStore: dict, write: bool=True):
        """
        Stores the generated data in a csv-file, creates a new one if it does not exist.\\
        
        ### Params:
            * paramsToStore: dict, keys -> features, values: list
        """
        print("Storing the generated data, ", end="")
        df = DataFrame(paramsToStore)
        if write:
            try:
                oldDF = read_csv(PATH+"/data/testGenerateData.csv")
                writeDF = concat([oldDF, df])
                writeDF.to_csv(PATH+"/data/testGenerateData.csv", index=False)
                print("added it to the file.")
            except:
                df.to_csv(PATH+"/data/testGenerateData.csv", index=False)
                print("created a new file and added it there.")


    def generateDataWithSim(self, simDuration: float=10, updateInterval: float=1, window: float=0.5):
        """
        ### NOTE should maybe store the average value from the last second?

        ### Params:
            * TODO
        """
        acceleration = [0]
        timeRan = 0 # seconds

        storePredictions = True
        paramsToStore = {"Time": [0],
                         "TTC": [0],
                         "DTO": [0],
                         "JERK": [0],
                         "Speed": [0],
                         "AngularSpeedX": [0],
                         "AngularSpeedY": [0],
                         "AngularSpeedZ": [0],
                         "COL": [0]}

        metrics = [" s", " s", " m", " m/s^3", " m/s", " m/s", " m/s", " m/s", ""]

        self.changeTimeAndWeather(6)

        lidar = ReadLidar(window, 35)

        intsPerSec = int(1//updateInterval)
        print(f"Updating {intsPerSec} times per second!")

        while True:
            self.sim.run(updateInterval) # NOTE can speed up the virtual time in the simulator
            timeRan += updateInterval

            if storePredictions and timeRan % 1 == 0:
                for sensor in self.ego.get_sensors(): # maybe use __dict__ to get to the sensor immediately
                    if sensor.name == "Lidar":
                        sensor.save(PATH + "/data/lidarUpdate.pcd") # TODO make this filename work on other PCs
                        paramsToStore["DTO"].append(lidar.updatedDTO)
                        break
                
                paramsToStore["Time"].append(timeRan)
                paramsToStore["Speed"].append(round(self.ego.state.speed, 3))
                acceleration.append((paramsToStore["Speed"][-1]-paramsToStore["Speed"][-2])/updateInterval)
                paramsToStore["JERK"].append(round(abs((acceleration[-1]-acceleration[-2])/updateInterval), 3)) # NOTE looks like jerk is always positive in the dataset
                paramsToStore["AngularSpeedX"].append(round(self.ego.state.angular_velocity.x, 3))
                paramsToStore["AngularSpeedY"].append(round(self.ego.state.angular_velocity.y, 3))
                paramsToStore["AngularSpeedZ"].append(round(self.ego.state.angular_velocity.z, 3))
                paramsToStore["TTC"].append(round(self.getTTC(paramsToStore["Speed"][-1], paramsToStore["DTO"][-1], paramsToStore["DTO"][-1], updateInterval), 3))
                paramsToStore["COL"].append(1 if self.actualCollisionTimeStamp > timeRan-1 else 0)

                ### Some nice information
                for (feature, value), metric in zip(paramsToStore.items(), metrics):
                    if "Angular" in feature: continue
                    val = f"{value[-1]}{metric}".ljust(15)
                    print(f"{feature}: {val}", end="")
                else:
                    print()

            if timeRan > simDuration:
                self.sim.stop()
                break

        if storePredictions:
            self.storeDataGenerated(paramsToStore, True)

if __name__ == "__main__":
    sim = GenerateData("sf")
    sim.generateDataWithSim(simDuration=10, updateInterval=0.5, window=1.0)











