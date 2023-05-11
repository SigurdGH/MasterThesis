import os
import sys
from environs import Env
import lgsvl
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
# from threading import Thread
from time import time
from pandas import DataFrame
import math

PATH = os.path.abspath(__file__)
PATH = PATH[:PATH.find("main")-1]
sys.path.insert(0, PATH)

from main.ML.Model import Predicter
from main.readLidar import ReadLidar
from main.generateTestingData.useGeneratedData import NewPredicter


# TODO
# Seems like positions are wrong when loading files to the scenariorunner.
# Possibly make something to transform the positions.
# Could maybe use 'easting' and 'northing' to get precise locations from xml

# TODO
# * Maybe use absolute value in stead of average jerk


# def on_collision(agent1, agent2, contact):
#     """
#     From deepscenario-toolset
#     """
#     name1 = agent1.__dict__.get('name')
#     name2 = agent2.__dict__.get('name') if agent2 is not None else "OBSTACLE"
#     print(f"{name1} collided with {name2} at {contact}")


def on_custom(agent, kind, context):
    # FRA GITHUB
    print(f"Agent: {agent}")
    print(f"Kind: {kind}")
    print(f"Context: {context}")
    if kind == "comfort":
        print(f"Comfort sensor callback: {context}")


def fromScenario(filename: str="", mode: int=0):
    """
    Run a scenario from a scenariofile.

    From github: https://github.com/Simula-COMPLEX/DeepScenario/tree/main/deepscenario-toolset#requirements
    """
    runner = lgsvl.scenariotoolset.ScenarioRunner()
    runner.load_scenario_file(scenario_filepath_or_buffer=filename)
    runner.connect_simulator_ads(simulator_port=8181, bridge_port=9090)
    runner.run(mode) # mode=0: disable ADSs; mode=1: enable ADSs


def plotData(speeds, accelerations, jerks, predictions, duration, interval, dto, ttc):
    x = [i for i in range(len(speeds))]

    # print(len(s), len(a), len(j), len(p), duration, interval, len(dto))

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(x, speeds, "-", label="Speed (m/s)")
    plt.plot(x, accelerations, "--", label="Acceleration (m/s^2)")
    plt.plot(x, jerks, ":", label="Jerk (m/s^3)")
    plt.plot(x, [i*10 for i in predictions], "-r", label="Predicted collision (0 or 10)")
    plt.xlabel(f"Time, delta = {interval} s, duration = {duration} s")
    plt.legend()
    plt.grid()
    plt.title("Speed, acceleration, jerk and collision prediction")

    plt.subplot(122)
    plt.plot(x, speeds, "-", label="Speed (m/s)")
    plt.plot(x, [i*10 for i in predictions], "-r", label="Predicted collision (0 or 10)")
    plt.plot(x, dto, "--", label="DTO (m)")
    plt.plot(x, ttc, "-", label="TTC (s)")
    plt.xlabel(f"Time, delta = {interval} s, duration = {duration} s")
    plt.legend()
    plt.grid()
    plt.title("Speed, DTO and collision prediction")

    plt.show()


# class P():
#     def __init__(self, filename) -> None:
#         self.model = Predicter()
#         self.model.loadModel(filename)
#         self.translate = {1: "You are going to crash!", 0: "Not crashing!"}

#     def predict(self, ttc=20, dto=20, jerk=0, speeds=[0,0,0,0,0,0], angular=[]):
#         """
#         Used to predict: [TTC, DTO, Jerk, road, scenario, speed1, speed2, speed3, speed4, speed5, speed6]

#         TODO
#         Make sure the values are correct.
#         Make it so prediction only takes from each second, if updateInterval is lower then 1.

#         Attribute[TTC]	Attribute[DTO]	Attribute[Jerk]	reward	road	strategy	scenario	speed1	speed2	speed3	speed4	speed5	speed6
#         100000.000000	6.434714	5.04	ttc	road2	random	rain_night	5.547	4.660	4.401	4.228	3.986	3.746
        
#         ### Params
#             ...

#         """
#         # x = np.array([20, 20, 1, 2, 1,1,2,3,4,5,6])
#         if len(angular) < 6:
#             x = [ttc, dto, jerk] + speeds
#         else:
#             x = [dto, jerk] + speeds + np.array(angular).flatten().tolist()
#         # print(f"Inne i predict: {x}")
#         xP, _ = self.model.preProcess(x)
#         # print(f"x: {x}\t\tprocessed: {xP[0]}")
#         prediction = self.model.predict(xP)[0]
#         # return self.translate[prediction]
#         return prediction
        



class Simulation():
    """
    Simulation class

    ### NOTE:
    The simulator program might need to be restarted if the map is changed.

    ### Params:
        * map: str, 'bg' for Borregas Avenue, 'sf' for San Francisco or 'ct' for Cube Town.
    """
    def __init__(self, map: str="bg") -> None:
        self.env = Env()
        self.sim = lgsvl.Simulator(self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host), self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        
        
        self.maps = {"sf": lgsvl.wise.DefaultAssets.map_sanfrancisco,
                     "bg": lgsvl.wise.DefaultAssets.map_borregasave,
                     "ct": lgsvl.wise.DefaultAssets.map_cubetown}
        self.changeMap(map)
        
        self.state = lgsvl.AgentState()
        self.spawns = self.sim.get_spawn()
        self.state.transform = self.spawns[0]

        # sp = self.spawns[0]
        # forward = lgsvl.utils.transform_to_forward(self.spawns[0])
        # right = lgsvl.utils.transform_to_right(self.spawns[0])
        # self.state.transform.position = self.spawns[0].position + 120 * forward
        # sp[0] += 120*lgsvl.utils.transform_to_forward(self.spawns[0])
        # self.state.transform = sp
        
        self.ego = self.sim.add_agent(self.env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.ego_lincoln2017mkz_apollo5), lgsvl.AgentType.EGO, self.state)
        self.actualCollisionTimeStamp = -1
        self.ego.on_collision(self.on_ego_collision)
        self.ego.on_custom(on_custom) # NOT WORKING
        
        self.controls = lgsvl.VehicleControl()
        
        self.previousCrash = None
        self.otherAgents = []
        self.distanceToObjects = []


    def on_ego_collision(self, agent1, agent2, contact):
        """
        From deepscenario-toolset.

        ### NOTE
        Sometimes a collision with a tree (++?) is not registered!
        """
        self.actualCollisionTimeStamp = self.sim.current_time
        name1 = agent1.__dict__.get('name')
        name2 = agent2.__dict__.get('name') if agent2 is not None else "OBSTACLE"
        print(f"{name1} collided with {name2} at {contact}, time: {round(self.actualCollisionTimeStamp, 2)}")


    def changeMap(self, map: str="bg"):
        """
        Change or load map.

        ### Params:
            * map: str, 'bg' for Borregas Avenue or 'sf' for San Francisco
        """
        try:
            if self.sim.current_scene ==  self.maps[map]:
                self.sim.reset()
            else:
                self.sim.load(self.maps[map])
        except KeyError as e:
            print(f"This map does not exist in this program. Error: {e}")
            

            # if self.selectedMap == self.maps[map]:

            # self.selectedMap = self.maps[map]
        
        # if map != "bg":
        #     self.selectedMap = lgsvl.wise.DefaultAssets.map_sanfrancisco

        # if self.sim.current_scene ==  self.selectedMap:
        #     self.sim.reset()
        # else:
        #     self.sim.load(self.selectedMap)


    def changeTimeAndWeather(self, time: int=0, rain: float=0.0, fog: float=0.0, damage: float=0.0):
        """
        Change weather and time in the simulation.
        Setting 'rain' also affects cloudiness and road wetness.
        Turns on the headlights if the time is between 19 and 7.

        ### Params:
            * time: int, 0-23 (Can also use datime for more specific adjustments)
            * rain: float, 0-1: 0.2 = 20% rain
            * fog: float, 0-1: 0.2 = 20% fog
            * damage: float, 0-1: 0.2 = 20% damage, road damage
        """
        if time >= 0 and time <= 23:
            self.sim.set_time_of_day(time)
        else:
            print("Time must be 0-23!")
        change = None
        if rain >= 0 and rain <= 1 and rain >= 0 and rain <= 1 and rain >= 0 and rain <= 1:
            change = lgsvl.WeatherState(rain=rain, fog=fog, wetness=rain, cloudiness=rain, damage=damage)
        else:
            print(f"The parameters sent in are not valid! You sent in: rain: {rain}, fog: {fog}, damage{damage}.")

        self.controls.headlights = 1 if time > 19 or time < 7 else 0
        # self.controls.windshield_wipers = True if rain > 0.1 else False # Wipers do not work :(

        self.sim.weather = change


    def changeScenario(self, scenario: str="sunny_day"):
        """
        Change time and weather regarding different scenarion.
        NOTE: The time is approximate and the 'raininess' is guessed as of now.

        ### Params:
            * scenario: str, four scenarios: rain_day, rain_night, sunny_day, sunny_night

        """
        scenarios = {
            "rain_day": {"time": 9, "rain": 0.8},
            "rain_night": {"time": 21, "rain": 0.8},
            "sunny_day": {"time": 9, "rain": 0},
            "sunny_night": {"time": 21, "rain": 0}}
        self.changeTimeAndWeather(time=scenarios[scenario]["time"], rain=scenarios[scenario]["rain"])


    def spawnNPCVehicle(self, npcType: str="Sedan", front: int=10, side: int=0, rotation: int=0, speed: int=0, followLane: bool=False):
        """
        Spawn an NPC car that will follow the closest lane.
        Also stops for traffic lights.

        ### Params:
            * vehicle: str, which vehicle to spawn (Sedan, SUV, Jeep, Hatchback, SchoolBus, BoxTruck)
            * front: int, how many meters the NPC will spawn in front of the EGO vehicle
            * side: int, how many meters the NPC will spawn to the right of the EGO vehicle
            * rotation: int, how many degress the NPC will rotate in regards to the EGO vehicle
            * speed: int, set the speed (m/s) of the NPC
        """
        vehicles = ["Sedan", "SUV", "Jeep", "Hatchback", "SchoolBus", "BoxTruck"]
        pedestrians = ["Bob", "Bill", "EntrepreneurFemale", "Howard", "Jun", "Johny", "Pamela", "Presley"] # More predestians exists
        npc = None
        if npcType in vehicles:
            # print("The vehicle chosen does not exist!")
            # return
        
            state = lgsvl.AgentState()

            forward = lgsvl.utils.transform_to_forward(self.spawns[0])
            right = lgsvl.utils.transform_to_right(self.spawns[0])
            state.transform.position = self.spawns[0].position + front * forward + side * right
            self.spawns[0].rotation.y += rotation
            state.transform.rotation = self.spawns[0].rotation
            
            npc = self.sim.add_agent(npcType, lgsvl.AgentType.NPC, state)
            if speed > 0 and followLane:
                npc.follow_closest_lane(True, speed)
            # npc.on_collision(self.on_collision)
        if npcType in pedestrians:
            npc = self.sim.add_agent(npcType, lgsvl.AgentType.PEDESTRIAN)
        self.otherAgents.append(npc)
        self.distanceToObjects.append(100000)    


    def useScenario(self, scene=1):
        """
        Different scenarios for simulating.\\
        ### Scene = 1:
        A simple scenario where a sedan spawns 20 meters in front of the ego vehicle with a set speed and the ego is cathing up.
        The ego vehicle is going to crash into it after some seconds.

        """
        print(f"Running scene number {scene}!")
        forward = lgsvl.utils.transform_to_forward(self.spawns[0])
        right = lgsvl.utils.transform_to_right(self.spawns[0])
        rot = self.spawns[0].rotation
        waypointsList = []

        # if self.sim.current_scene != self.maps.get(map, ""): self.changeMap(map)

        def runWithWaypoints(waypointsList):
            for index, waypoints in enumerate(waypointsList):
                if len(waypoints) > 0:
                    # waypoints = [lgsvl.DriveWaypoint(position=self.spawns[0].position + f * forward + r * right, speed=5, angle=rot) for f, r in directions]
                    self.otherAgents[index].follow(waypoints, loop=False)
                    self.otherAgents[index].on_waypoint_reached(self.onReach)
                    # self.otherAgents[0].change_lane(False)
            
        if scene == 1: # Car driving in front
            self.spawnNPCVehicle("Sedan", 20, 0.5, 0, 8, True)
            self.controls.throttle = 0.5
        elif scene == 2: # Left lane, overtake, change to right lane, continue
            self.spawnNPCVehicle("Sedan", -5, -4, 0, 8, True)
            directions = [(0, -4), (10, -4), (30,-4), (33,0), (50,0)]
            waypointsList.append([lgsvl.DriveWaypoint(position=self.spawns[0].position + f * forward + r * right, speed=5, angle=rot) for f, r in directions])
            runWithWaypoints(waypointsList)
            self.controls.throttle = 0.2
        elif scene == 3: # Driving between two vehicles
            self.spawnNPCVehicle("SUV", 10, 0, 0, 10, True)
            self.spawnNPCVehicle("BoxTruck", 8, -4, 0, 9, True)
            self.controls.throttle = 0.4
        elif scene == 4: # Car driving in front and changing lane
            self.spawnNPCVehicle("Hatchback", 10, 0, 0, 8, True)
            directions = [(42, 0), (48, -4), (60, -4)]
            waypointsList.append([lgsvl.DriveWaypoint(position=self.spawns[0].position + f * forward + r * right, speed=5, angle=rot) for f, r, i in directions])
            runWithWaypoints(waypointsList)
            self.controls.throttle = 0.4
        elif scene == 5: # Meeting car at intersection
            self.spawnNPCVehicle("SUV", 145, 32, 218)
            directions = [(145, 32, 2), (100, -10, 0)]
            waypointsList.append([lgsvl.DriveWaypoint(position=self.spawns[0].position + f * forward + r * right, speed=5, angle=rot, idle=i) for f, r, i in directions])
            runWithWaypoints(waypointsList)
            self.controls.throttle = 0.4

        # elif scnene == ... # With pedestrians
        #     self.otherAgents.append(self.sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN))
        #     # directions = [(70, 3), (70, -10), (70, 3)]
        #     directions = [(5, 3), (5, -10)]
        #     waypointsList.append([lgsvl.WalkWaypoint(position=self.spawns[0].position + f * forward + r * right, idle=0, speed=1) for f, r in directions])
        #     runWithWaypoints(waypointsList)
        #     self.controls.throttle = 0.
        #     npc = self.sim.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN)
        #     waypoints = [
        #         lgsvl.WalkWaypoint(self.spawns[0], 1, 0),
        #         lgsvl.WalkWaypoint(self.spawns[0], 2, 0),
        #         lgsvl.WalkWaypoint(self.spawns[0], 1, 0),
        #     ]
        #     npc.follow(waypoints, loop=True)
        #     self.sim.add_random_agents(lgsvl.AgentType.PEDESTRIAN)


    def onReach(self, agent, index):
        """
        Used by an NPC when a waypoint is reched.\\
        NPC.on_waypoint_reached(self.onReach)
        """
        # print(agent)
        print(f"{agent.name} has reached waypoint {index}")
        # print(args)
        # agent.change_lane(False)
        # self.otherAgents[0].change_lane(False)


    def isColliding(self, lastCollision: float=0.0, now: float=0.0):
        """
        Turns on warning hazards for 5 seconds when a crash is predicted and applies the brakes.

        NOTE: Could maybe make it more abstract in 'run' by implementing this?
        if predictions[-1] and predictions[-1] != predictions[-2]:
            lastCollision = timeRan

        self.isColliding(lastCollision, timeRan, predictions[-1], predictions[-2])
        """
        if now-lastCollision > 5:
            self.controls.braking = 0
            self.controls.turn_signal_left = False
            self.controls.turn_signal_right = False
        else:
            self.controls.braking = 1
            self.controls.turn_signal_left = True
            self.controls.turn_signal_right = True


    def getDTOsFromCoordinates(self):
        """
        Calculate the distance between the EGO vehicle and all other NPCs by looking at the 
        coordinates for all NPCs in the simulator.
        """
        def getDistance(v1, v2):
            return np.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)

        def getPos(agent):
            return (agent.transform.position.x, agent.transform.position.y, agent.transform.position.z)

        egoPos = getPos(self.ego)
        for i, agent in enumerate(self.otherAgents):
            # print(f"Distance to NPC{i}: ", end="")
            self.distanceToObjects[i] = round(getDistance(egoPos, getPos(agent)), 6)
    

    def getTTC(self, currentSpeed: float=10, previousDistance: float=0, currentDistance: float=0, timeDifference: float=1):
        """
        Calculate time to collision by looking at EGO speed and other cars speed and direction.
        ### TODO
        Calculate TTO when both vehicles are moving
        """
        ttc = 100000
        # dist/speed
        # print(f"distance prev: {previousDistance}, currentDistance: {currentDistance}")
        if currentDistance < 100 and currentSpeed > 1:
            ttc = currentDistance / currentSpeed
            # print(ttc)
        
        ttc = 50 if ttc > 50 else ttc
        return ttc


    def writeParameters(self, paramsToStore: list[list], write: bool=True):
        # with open("additionalData.csv", "a")
        df = DataFrame(columns=["Attribute[TTC]", "Attribute[DTO]", "Attribute[Jerk]", "speed1", "speed2", "speed3", "speed4", "speed5", "speed6", "av1x", "av1y", "av1z", "av2x", "av2y", "av2z", "av3x", "av3y", "av3z", "av4x", "av4y", "av4z", "av5x", "av5y", "av5z", "av6x", "av6y", "av6z", "Predicted[COL]", "Attribute[COL]"]) 
        # print(f"cols: {len(df.columns)}")
        for line in paramsToStore:
            # print(f"line: {len(line)}")
            df.loc[len(df)] = line
        if write:
            df.to_csv("MasterThesis/data/additionalData.csv", index=False)


    @staticmethod
    def getSpeedOfObject(distance0: float, distance: float, preSpeed: float, speed: float, interval: float) -> float:
        """
        Calculates the speed of an object in front of the EGO vehicle.\\
        #### Formula:
            * distanceEgo = interval * (speed + preSpeed)/2
            * distanceObject = distanceEgo + distance - distance0
            * speedObject = distanceObject / interval
            * -> v_object = ((t * (v_ego_1 + v_ego_1)/2) + d_1 - d_0) / t
        ### Params:
            * distance0: float, distance (m) to the object one "interval" ago
            * distance: float, distance (m) to the object one "interval" ago
            * preSpeedA: float, speed (m/s) of the EGO vehicle one "interval" ago
            * speed: float, current speed (m/s) of the EGO vehicle
            * interval: float, time (s) interval between each update
        
        ### Returns:
            * float, speed (m/s) of the object in front of the EGO vehicle
        """
        # distanceEgo = interval * speed + 0.5 * acc * interval**2
        # distanceEgo = interval * (speed + preSpeed)/2
        # distanceObject = distanceEgo + distance - distance0
        # speedObject = distanceObject / interval
        return round(((interval * (speed + preSpeed)/2) + distance - distance0) / interval, 3)


    @staticmethod
    def calculateTTC(preDistance: float, distance: float, preSpeed: float, speed: float, accA: float, interval: float) -> float:
        """
        Calculate time to collision when both objects can be on the move.

        ### Formulas
        d = vt + 0.5at^2\\
        0.5at^2 + vt + (-d) = 0
        --> t = (-v +- sqrt(v^2 - 4*0.5*a*d)) / (2 * 0.5 a)

        ### NOTE
        Maybe not necessary to use acceleration / it is more precise to not use it

        ### NOTE
        If the distance is high enough and the acceleration is negative enough, 
        the NPC vehicle is "calculated" to have negative speed because the acceleration is constant

        ### Params:
            * preDistance: float, distance (m) to the object one "interval" ago
            * distance: float, current distance (m) to the object
            * preSpeedA: float, speed (m/s) of the EGO vehicle one "interval" ago
            * speed: float, current speed (m/s) of the EGO vehicle
            * acc: float, the EGO vehicle's acceleration (m/s^2)
            * interval: float, time (s) interval between each update
        
        ### Returns:
            * ttc: float, time (s) to collision
        """
        NO_COLLISION = 50

        if distance >= 100:
            return NO_COLLISION
        if preDistance - distance > 3 * speed:
            # print("\tNEW OBJECT")
            return NO_COLLISION

        speedObject = Simulation.getSpeedOfObject(preDistance, distance, speed, preSpeed, interval)
        relativeSpeed = speed - speedObject
        relativeAcceleration = 0 # accA - accB # Maybe not necessary with acceleration of B
        # print(f"\nEGO: {round(speed, 2)}, object: {round(speedObject, 2)}, speed diff: {round(relativeSpeed, 2)}, dto: {distance}")
        
        if relativeAcceleration == 0:
            if speed <= speedObject:
                return NO_COLLISION
            ttc = distance / relativeSpeed
            return ttc if ttc <= 50 else 50
        
        # num = relativeSpeed**2 - 4 * 0.5 * relativeAcceleration * (-distance)
        # # print(f"(-{relativeSpeed} +- sqrt({relativeSpeed}^2 - 4*0.5*{relativeAcceleration}*(-{distance}))) / 2 * 0.5 {relativeAcceleration}")
        # if num >= 0:
        #     t_add = (-relativeSpeed + math.sqrt(num)) / (2*0.5*relativeAcceleration)
        #     t_sub = (-relativeSpeed - math.sqrt(num)) / (2*0.5*relativeAcceleration)
        #     if t_add > 0 and t_add < 50:
        #         return t_add
        #     elif t_sub > 0 and t_sub < 50:
        #         return t_sub
        # return NO_COLLISION # Not going to collide


    def runSimulation(self, 
                      simDuration: float=10, 
                      updateInterval: float=0.5, 
                      window: float=0.5, 
                      model: str="Classifier", 
                      runScenario: int=0, 
                      plotting: bool=True, 
                      storePredictions: bool = False,
                      useGeneratedData: bool=True):
        """
        Run a simulation in LGSVL (OSSDC-SIM).

        ### Params:
            * simDuration: float, time (seconds) for simulation duration
            * updateInterval: float, time (seconds) between each data logging
            * window: float, distance (meters) left/right the algorithm should look for obstacles
            * model: str, which model the predicter class should use
            * runScenario: int, if 0, the car can be driven with the keyboard, otherwise a scenario
            * plotting: bool, plot speed, acceleration, jerk, predictions and DTO after the simulation
        """        
        ### Variables
        pastImportance = 4
        ttcList = [50] # seconds
        dtoList = [100] # meters
        speeds = [0] # m/s
        acceleration = [0] # m/s^2
        jerk = [0] # m/s^3
        predictions = [0] * int(5 // updateInterval) # bool, 0 or 1
        angular = [[0,0,0]]  # m/s, m/s, m/s
        angularX = [0] # m/s
        angularY = [0] # m/s
        angularZ = [0] # m/s
        timeRan = 0 # seconds
        lastCollision = -100 # seconds
        printingInfo = [("TTC: ", " s"), ("DTO: ", " m"), ("JERK: ", " m/s^3"), ("Speed: ", " m/s"), ("Time: ", " s")]
        
        # df = DataFrame(columns=["Attribute[TTC]", "Attribute[DTO]", "Attribute[Jerk]", "speed1", "speed2", "speed3", "speed4", "speed5", "speed6", "av1x", "av1y", "av1z", "av2x", "av2y", "av2z", "av3x", "av3y", "av3z", "av4x", "av4y", "av4z", "av5x", "av5y", "av5z", "av6x", "av6y", "av6z", "Predicted[COL]", "Attribute[COL]"]) 
        paramsToStore = []

        ### Scenarios
        # self.changeScenario("sunny_night")
        self.changeTimeAndWeather(6)
        if runScenario > 0:
            print(f"starting simulation with scenario {runScenario}...")
            self.useScenario(runScenario)
            # self.controls.throttle = 0.2
            # self.spawnNPCVehicle("Sedan", 30, 0.5, 10, True)
        else:
            print("Driving with keyboard!")
            self.spawnNPCVehicle("Sedan", 10, 0, 0, 10, True)
        
        ### Classes
        # pred = P(model) # Import predictor and load model with new predicts?
        if useGeneratedData:
            predicter = NewPredicter.loadModel(model)
            predictions = [0] * 4
        else:
            predicter = Predicter() 
            predicter.loadModel(model)
        lidar = ReadLidar(window, 35)

        intsPerHalfSec = int(0.5//updateInterval) # NOTE this might not work as intended with updateInterval != 0.5
        print(f"Updating {intsPerHalfSec*2} times per second!")

        oldControls = copy(self.controls.__dict__)
        self.controls.headlights = 0
        self.ego.apply_control(self.controls, False)
        while True:
            self.sim.run(updateInterval) # NOTE can speed up the virtual time in the simulator
            timeRan += updateInterval
            self.ego.get_sensors()[2].save(PATH + "/data/lidarUpdate.pcd")
            dtoList.append(lidar.updatedDTO)
            # Maybe use this to check distances in comparison with the lidar
            # self.getDTOs()
            speeds.append(round(self.ego.state.speed, 3))
            acceleration.append((speeds[-1]-speeds[-2])/updateInterval)
            jerk.append(round(abs((acceleration[-1]-acceleration[-2])/updateInterval), 3)) # NOTE looks like jerk is always positive in the dataset

            ### Angular
            angular.append([round(self.ego.state.angular_velocity.x, 3), round(self.ego.state.angular_velocity.y, 3), round(self.ego.state.angular_velocity.z, 3)])
            angularX.append(round(self.ego.state.angular_velocity.x, 3))
            angularY.append(round(self.ego.state.angular_velocity.y, 3))
            angularZ.append(round(self.ego.state.angular_velocity.z, 3))

            ### Old TTC
            # ttcList.append(round(self.getTTC(speed, dtoList[-2], dtoList[-1], updateInterval), 3))
            ttcList.append(self.calculateTTC(dtoList[-2], dtoList[-1], speeds[-1], speeds[-2], acceleration[-1], updateInterval))

            ## Some nice information
            stuff = [round(ttcList[-1], 2), round(dtoList[-1], 2), round(np.average(jerk[-(6):]), 2), round(speeds[-1], 3), round(self.sim.current_time, 1)]
            for info, value in zip(printingInfo, stuff):
                print(f"{info[0]}{value}{info[1]}".ljust(22), end="")
            else:
                print()
            # print(f"TTC: {round(ttcList[-1], 2)} s \t DTO: {round(dtoList[-1], 2)} Jerk: {round(np.average(jerk[-(6):]), 2)} m/s^3\t Speed: {round(self.ego.state.speed, 3)} m/s \t Time: {round(self.sim.current_time, 1)} s")

            ### Evasive action
            # if dtoList[-1] < 15:
            #     evasive = lidar.getEvasiveAction()
            #     evasiveDict = {1: "LEFT", -1: "RIGHT"}
            #     print(f"Turn {evasiveDict[evasive]} to avoid a potential collision!")
                # self.controls.steering = -evasive/4
                # self.controls.braking = 1
            # else:
            #     self.controls.steering = 0

            ### Starts the collision prediction
            if (timeRan//updateInterval >= pastImportance) or (len(speeds) > 5 // updateInterval):
            # if len(speeds) > 2 // updateInterval:
                # predictions.append(pred.predict(ttc=ttcList[-1], dto=dtoList[-1], jerk=np.average(jerk[-(6):]), speeds=speeds[-(6*intsPerHalfSec)::intsPerHalfSec]))
                if useGeneratedData:
                    row = ttcList[-pastImportance:] + dtoList[-pastImportance:] \
                        + jerk[-pastImportance:] + speeds[-pastImportance:] \
                            + angularX[-pastImportance:] + angularY[-pastImportance:] \
                                + angularZ[-pastImportance:]
                    # print(row)
                    predictions.append(predicter.predict(predicter.preProcess(row)))
                else:
                    predictions.append(predicter.predict(dto=dtoList[-1], 
                                                    jerk=round(np.average(jerk[-(6*intsPerHalfSec)::intsPerHalfSec]), 3), 
                                                    speeds=speeds[-(6*intsPerHalfSec)::intsPerHalfSec], 
                                                    angular=angular[-(6*intsPerHalfSec)::intsPerHalfSec]))

                if storePredictions:
                    # TODO Make it so the last x before an actual collision also is 1?
                    actualCollision = 1 if self.actualCollisionTimeStamp > timeRan-updateInterval else 0
                    if len(paramsToStore) > 0: paramsToStore[-1][-1] = actualCollision
                    paramsToStore.append([ttcList[-1], dtoList[-1], round(np.average(jerk[-(6*intsPerHalfSec)::intsPerHalfSec]), 3)] + [i for i in speeds[-(6*intsPerHalfSec)::intsPerHalfSec]] + [i for xyz in angular[-(6*intsPerHalfSec)::intsPerHalfSec] for i in xyz] + [predictions[-1], None])


            ### Check if a collision has been predicted
            if predictions[-1] and predictions[-1] != predictions[-2]:
                print("A COLLISION IS GOING TO HAPPEN!")
                lastCollision = timeRan

            ### Turns on hazards and applies the brakes
            self.isColliding(lastCollision, timeRan)

            ### Only applies new controls if something has changed
            if self.controls.__dict__ != oldControls and bool(runScenario):
                oldControls = copy(self.controls.__dict__)
                self.ego.apply_control(self.controls, True)

            if timeRan > simDuration:
                self.sim.stop()
                break

        if storePredictions:
            self.writeParameters(paramsToStore, True)

        if plotting:
            print(len(speeds))
            print(len(acceleration))
            print(len(jerk))
            print(len(predictions))
            # print(len(simDuration))
            # print(len(updateInterval))
            print(len(dtoList))
            print(len(ttcList))

            plotData(speeds, acceleration, jerk, predictions, simDuration, updateInterval, dtoList, ttcList)


if __name__ == "__main__":
    # file = "C:/MasterFiles/DeepScenario/deepscenario-dataset/greedy-strategy/reward-dto/road3-sunny_day-scenarios/0_scenario_8.deepscenario"
    sim = Simulation("sf")
    # # sim.runSimulation(30, 1, 0.5, "Classifier", 5, False) # "xgb_2_582-11-16-201"
    sim.runSimulation(simDuration=15,
                      updateInterval=0.5,
                      window=1.0,
                      model = "MLPClassifierWithGeneratedData",
                    #   model = "xgb_gen_438-6-20-10",
                    #   model="xgb_2_582-11-16-201", # NOTE må bruke ny modell
                      runScenario=0,
                      plotting=True,
                      storePredictions=False,
                      useGeneratedData=True)
    
