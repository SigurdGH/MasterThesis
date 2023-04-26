import os
import sys
import lgsvl
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
        Spawns random NPCs, both pedestrians and vehicles that are roaming the map.\\
        NPCs spawn around the area the EGO vehicle is located.
        
        ### Params:
            * amountVehicles: int, number of vehicles to spawn
            * amountPedestrians: int, number of pedestrians to spawn
        """
        for _ in range(amountVehicles):
            self.sim.add_random_agents(lgsvl.AgentType.NPC)
        for _ in range(amountPedestrians):
            self.sim.add_random_agents(lgsvl.AgentType.PEDESTRIAN)


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
        Run the simulation and driving with the keyboard.\\
        Generate data from each second of running.
        Stores the parameters Time, TTC, DTO, JERK, Speed, AngularSpeedX, AngularSpeedY, AngularSpeedZ, COL in a csv-file.

        ### Params:
            * simDuration: float, time (seconds) for simulation duration
            * updateInterval: float, time (seconds) between each data logging
            * window: float, distance (meters) left/right the algorithm should look for obstacles
        """
        acceleration = [0]
        timeRan = 0 # seconds
        timeBetweenLogging = 1 # seconds

        storePredictions = True
        paramsToStore = {"Time": [0],
                         "TTC": [0],
                         "DTO": [0],
                         "JERK": [0],
                         "Speed": [0],
                         "asX": [0],
                         "asY": [0],
                         "asZ": [0],
                         "COL": [0]}

        metrics = [" s", " s", " m", " m/s^3", " m/s", " m/s", " m/s", " m/s", ""]

        self.changeTimeAndWeather(6)

        lidar = ReadLidar(window)

        intsPerSec = int(1//updateInterval)
        print(f"Updating {intsPerSec} times per second!")

        while True:
            self.sim.run(updateInterval) # NOTE can speed up the virtual time in the simulator
            timeRan += updateInterval

            if timeRan % timeBetweenLogging == 0: # Once every second
                self.ego.get_sensors()[2].save(PATH + "/data/lidarUpdate.pcd")
                paramsToStore["DTO"].append(lidar.updatedDTO)
                paramsToStore["Time"].append(timeRan)
                paramsToStore["Speed"].append(round(self.ego.state.speed, 3))
                acceleration.append((paramsToStore["Speed"][-1]-paramsToStore["Speed"][-2])/timeBetweenLogging)
                paramsToStore["JERK"].append(round(abs((acceleration[-1]-acceleration[-2])/timeBetweenLogging), 3))
                paramsToStore["asX"].append(round(self.ego.state.angular_velocity.x, 3))
                paramsToStore["asY"].append(round(self.ego.state.angular_velocity.y, 3))
                paramsToStore["asZ"].append(round(self.ego.state.angular_velocity.z, 3))
                if len(paramsToStore["DTO"]) > 2:
                    paramsToStore["TTC"].append(round(self.calculateTTC(paramsToStore["DTO"][-2], 
                                                                        paramsToStore["DTO"][-1], 
                                                                        paramsToStore["Speed"][-1], 
                                                                        paramsToStore["Speed"][-2], 
                                                                        acceleration[-1], 
                                                                        timeBetweenLogging), 3))
                else:
                    paramsToStore["TTC"].append(50)
                paramsToStore["COL"].append(1 if self.actualCollisionTimeStamp > timeRan-1 else 0)

                ### Some nice information to the console
                for (feature, value), metric in zip(paramsToStore.items(), metrics):
                    if "as" in feature: continue
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
    sim.spawnRandomNPCs(amountVehicles=30, amountPedestrians=10)
    sim.generateDataWithSim(simDuration=60, updateInterval=0.5, window=1.0)











