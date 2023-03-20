from environs import Env
import lgsvl
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from ML.Model import Predicter
from threading import Thread

# TODO
# Seems like positions are wrong when loading files to the scenariorunner.
# Possibly make something to transform the positions.
# Could maybe use 'easting' and 'northing' to get presice locations from xml

# TODO
# * Maybe use something else to get DTO
#     Also check if NPC is in front of EGO
# * Maybe use absolute value in stead of average jerk


def on_collision(agent1, agent2, contact):
    """
    From deepscenario-toolset
    """
    name1 = agent1.__dict__.get('name')
    name2 = agent2.__dict__.get('name') if agent2 is not None else "OBSTACLE"
    print("{} collided with {} at {}".format(name1, name2, contact))


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


def plotData(s, a, j, p, duration, interval, dto):
    x = [i for i in range(len(s))]

    # print(len(s), len(a), len(j), len(p), duration, interval, len(dto))

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(x, s, "-", label="Speed (m/s)")
    plt.plot(x, a, "--", label="Acceleration (m/s^2)")
    plt.plot(x, j, ":", label="Jerk (m/s^3)")
    plt.plot(x, [i*10 for i in p], "-r", label="Predicted collision (0 or 10)")
    plt.xlabel(f"Time, delta = {interval} s, duration = {duration} s")
    plt.legend()
    plt.grid()
    plt.title("Speed, acceleration, jerk and collision prediction")

    plt.subplot(122)
    plt.plot(x, s, "-", label="Speed (m/s)")
    plt.plot(x, [i*10 for i in p], "-r", label="Predicted collision (0 or 10)")
    plt.plot(x, dto, "--", label="DTO (m)")
    plt.xlabel(f"Time, delta = {interval} s, duration = {duration} s")
    plt.legend()
    plt.grid()
    plt.title("Speed, DTO and collision prediction")

    plt.show()


class P():
    def __init__(self, filename) -> None:
        self.model = Predicter()
        self.model.loadModel(filename)
        self.translate = {1: "You are going to crash!", 0: "Not crashing!"}

    def predict(self, ttc=20, dto=20, jerk=0, speeds=[0,0,0,0,0,0], angular=[]):
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
        # x = np.array([20, 20, 1, 2, 1,1,2,3,4,5,6])
        if len(angular) < 6:
            x = [ttc, dto, jerk] + speeds
        else:
            x = [ttc, dto, jerk] + speeds + np.array(angular).flatten().tolist()
        # print(f"Inne i predict: {x}")
        xP, _ = self.model.preProcess(x)
        # print(f"x: {x}\t\tprocessed: {xP[0]}")
        prediction = self.model.predict(xP)[0]
        # return self.translate[prediction]
        return prediction
        



class Simulation():
    """
    Simulation class

    ### Params:
        map: str, 'bg' for Borregas Avenue or 'sf' for San Francisco
    """
    def __init__(self, map: str="bg") -> None:
        self.env = Env()
        self.sim = lgsvl.Simulator(self.env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host), self.env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
        
        self.changeMap(map)
        
        self.state = lgsvl.AgentState()
        self.spawns = self.sim.get_spawn()
        self.state.transform = self.spawns[0]
        
        self.ego = self.sim.add_agent(self.env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.ego_lincoln2017mkz_apollo5), lgsvl.AgentType.EGO, self.state)
        self.ego.on_collision(on_collision)
        self.ego.on_custom(on_custom) # NOT WORKING
        
        self.controls = lgsvl.VehicleControl()
        
        # TODO log crashes with the plot
        self.previousCrash = None
        self.otherAgents = []
        self.distanceToObjects = []


    def changeMap(self, map: str="bg"):
        """
        Change or load map.

        ### Params:
            * map: str, 'bg' for Borregas Avenue or 'sf' for San Francisco
        """
        self.selectedMap = lgsvl.wise.DefaultAssets.map_borregasave
        if map != "bg":
            self.selectedMap = lgsvl.wise.DefaultAssets.map_sanfrancisco

        if self.sim.current_scene ==  self.selectedMap:
            self.sim.reset()
        else:
            self.sim.load(self.selectedMap)


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
        # self.controls.windshield_wipers = True if rain > 0.1 else False # Wipers does not work :(

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


    def spawnNPCVehicle(self, vehicle: str="Sedan", front: int=10, side: int=0, speed: int=0):
        """
        Spawn an NPC car that will follow the closest lane.
        Also stops for traffic lights.

        ### Params:
            * vehicle: str, which vehicle to spawn (Sedan, SUV, Jeep, Hatchback, SchoolBus, BoxTruck)
            * front: int, how many meters the NPC will spawn in front of the EGO vehicle
            * side: int, how many meters the NPC will spawn to the right of the EGO vehicle
            * speed: int, set speed of NPC in m/s
        """
        vehicles = ["Sedan", "SUV", "Jeep", "Hatchback", "SchoolBus", "BoxTruck"]
        if vehicle not in vehicles:
            print("The vehicle chosen does not exist!")
            return
        
        state = lgsvl.AgentState()

        forward = lgsvl.utils.transform_to_forward(self.spawns[0])
        right = lgsvl.utils.transform_to_right(self.spawns[0])
        state.transform.position = self.spawns[0].position + front * forward + side * right
        state.transform.rotation = self.spawns[0].rotation
        
        npc = self.sim.add_agent(vehicle, lgsvl.AgentType.NPC, state)
        npc.follow_closest_lane(True, speed)

        self.otherAgents.append(npc)
        self.distanceToObjects.append(100000)


    def simpleScenario(self):
        """
        A simple scenario where a sedan spawns 20 meters in front of the ego vehicle with a set speed and the ego is cathing up.
        The ego vehicle is going to crash into it after some seconds.
        """
        self.spawnNPCVehicle("Sedan", 20, 0, 8)
        self.controls.throttle = 0.6
        # controls.headlights = 1
        # self.ego.apply_control(self.controls, True)
        # return controls


    def isColliding(self, lastCollision: float=0.0, now: float=0.0):
        """
        Turns on warning hazards for 5 seconds when a crash is predicted.

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


    def getDTOs(self):
        """
        Calculate the distance between the EGO vehicle and all other NPCs.
        """
        def getDistance(v1, v2):
            return np.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)

        def getPos(agent):
            return (agent.transform.position.x, agent.transform.position.y, agent.transform.position.z)

        # print("EGO: ", end="")
        egoPos = getPos(self.ego)
        # print(egoPos[0])
        for i, agent in enumerate(self.otherAgents):
            # print(f"Distance to NPC{i}: ", end="")
            # print(getDistance(egoPos, getPos(agent)))
            self.distanceToObjects[i] = round(getDistance(egoPos, getPos(agent)), 6)
            # self.distanceToObjects[i] = np.sqrt(np.power(self.ego.transform.position.x-agent.transform.position.x, 2)+\
            #                                     np.power(self.ego.transform.position.y-agent.transform.position.y, 2)+\
            #                                         np.power(self.ego.transform.position.z-agent.transform.position.z, 2))


    def runSimulation(self, simDuration: float=10, updateInterval: float=1, plotting: bool=True):
        """
        Run simulation in LGSVL (OSSDC-SIM).

        ### Params:
            * simDuration: float, time (seconds) for simulation duration
            * updateInterval: float, time (seconds) between each data logging
            * plotting: bool, plot speed, acceleration, jerk, predictions and DTO after the simulation
        """        
        
        # self.changeTimeAndWeather(time=15, rain=0.4, fog=0.6, damage=0.2)
        self.changeScenario("sunny_night")
        # self.spawnNPCVehicle("Sedan", 10, speed=0)
        # self.spawnNPCVehicle("SchoolBus", 10, -10)
        # self.spawnNPCVehicle("BoxTruck", 20, speed=0)
        # self.spawnNPCVehicle("Sedan", 20, -3,speed=22)
        # self.otherAgents[0].follow_closest_lane(True, 5)

        self.simpleScenario()
        
        self.controls.headlights = 0

        # Both of these are seconds
        timeRan = 0
        lastCollision = -100

        intsPerSec = int(1//updateInterval)
        print(f"Updating {intsPerSec} times per second!")

        dtoList = [0]
        speeds = [0]
        acceleration = [0]
        jerk = [0]
        predictions = [0] * int(5 // updateInterval)
        
        angular = [[0,0,0]]

        pred = P("Classifier")
        # pred = P("MLPClassifier_1952-26-51-181") # ttc, dto, jerk, speed1-6, a1x, a1y, a1z, a2x...


        # def forThreading(ttc, jerk, speeds):
        #     self.getDTOs()
        #     dtoList.append(self.distanceToObjects[0]-4.77)
        #     predictions.append(pred.predict(ttc, dtoList[-1], jerk, speeds))

        #     if predictions[-1] and predictions[-1] != predictions[-2]:
        #         lastCollision = timeRan
        #     self.isColliding(lastCollision, timeRan)

        oldControls = copy(self.controls.__dict__)
        self.ego.apply_control(self.controls, True)
        # t = None
        notSaved = False
        while True:
            
            self.sim.run(updateInterval)
            timeRan += updateInterval

            # if timeRan > 1 and not notSaved:
            #     notSaved = True
            #     for sensor in self.ego.get_sensors():
            #         # print(sensor.name, end=", ")
            #         # print(sensor.__dict__)
            #         # print(sensor.enabled)
            #         if sensor.name == "Lidar":
            #             print("Saving lidar")
            #             sensor.save("C:/MasterFiles/MasterThesis/data/lidar2.pcd")
            #             print(f"min_distance: {sensor.min_distance}")
            #             print(f"max_distance: {sensor.max_distance}")
            #             print(f"rays: {sensor.rays}")
            #             print(f"rotations: {sensor.rotations}") # rotation frequency, Hz
            #             print(f"measurements: {sensor.measurements}") # = j["measurements"]  # how many measurements each ray does per one rotation
            #             print(f"fov: {sensor.fov}")
            #             print(f"angle: {sensor.angle}")
            #             print(f"compensated: {sensor.compensated}")



            # if isinstance(t, Thread):
            #     t.join()
            # t = Thread(target=self.sim.run, args=updateInterval)
            # t.start()

            self.getDTOs()

            # print(f"Angular velocity: x: {self.ego.state.angular_velocity.x},  y: {self.ego.state.angular_velocity.y},  z: {self.ego.state.angular_velocity.z}")
            ##
            dtoList.append(self.distanceToObjects[0]-4.77) # NOTE 4.77 is probably the distance between a position and length of a vehicle
            speed = self.ego.state.speed
            speeds.append(round(speed, 3))
            acceleration.append((speeds[-1]-speeds[-2])/updateInterval)
            jerk.append((acceleration[-1]-acceleration[-2])/updateInterval)

            # Angular
            angular.append([round(self.ego.state.angular_velocity.x, 3), round(self.ego.state.angular_velocity.y, 3), round(self.ego.state.angular_velocity.z, 3)])

            # print(f"Speed: {round(speeds[-1], 2)} m/s")
            # print(f"Acceleration: {round(acceleration[-1], 2)} m/s^2")
            # print(f"Jerk: {round(jerk[-1], 2)} m/s^3")
            # print(self.distanceToObjects[0], 5 // updateInterval)
            if len(speeds) > 5 // updateInterval:
                # t = Thread(target=forThreading, args=(20, np.average(jerk[-(6):]), speeds[-(6*intsPerSec)::intsPerSec]))
                # t.start()
                predictions.append(pred.predict(dto=dtoList[-1], jerk=np.average(jerk[-(6):]), speeds=speeds[-(6*intsPerSec)::intsPerSec]))
                # predictions.append(pred.predict(dto=dtoList[-1], jerk=np.average(jerk[-(6):]), speeds=speeds[-(6*intsPerSec)::intsPerSec], angular=angular[-(6*intsPerSec)::intsPerSec]))
            
            # NOTE: can maybe make it more abstract
            if predictions[-1] and predictions[-1] != predictions[-2]:
                lastCollision = timeRan
            self.isColliding(lastCollision, timeRan)#, predictions[-1], predictions[-2])
            
            if self.controls.__dict__ != oldControls: # Only applies new controls if something has changed
                oldControls = copy(self.controls.__dict__)
                self.ego.apply_control(self.controls, True)

            if timeRan > simDuration:
                # if isinstance(t, Thread):
                #     t.join()
                self.sim.stop()
                break

        if plotting:
            plotData(speeds, acceleration, jerk, predictions, simDuration, updateInterval, dtoList)


if __name__ == "__main__":
    # file = "C:/MasterFiles/DeepScenario/deepscenario-dataset/greedy-strategy/reward-dto/road3-sunny_day-scenarios/0_scenario_8.deepscenario"
    
    sim = Simulation("sf")
    sim.runSimulation(20, 1)
    # plotData(s,a,j)




# s = ScenarioRunner()
# s.load_scenario_file(scenario_filepath_or_buffer=file)
# print(s.get_entities_info())

# s = lgsvl.scenariotoolset.ScenarioRunner()
# s.load_scenario_file(scenario_filepath_or_buffer=file)
# # print(s.get_scene_by_timestep(timestep=21))

# for i in range(7): print(s.get_scene_by_timestep(timestep=i))

# NOTE Runs the simulator from sce
# runner = lgsvl.scenariotoolset.ScenarioRunner()
# runner.load_scenario_file(scenario_filepath_or_buffer=file)
# # runner.load_scenario_file(scenario_filepath_or_buffer='./deepscenario/overtake.deepscenario')
# runner.connect_simulator_ads(simulator_port=8181, bridge_port=9090)
# runner.run(mode=0) # mode=0: disable ADSs; mode=1: enable ADSs

# pred = P("Classifier")
# s = [10, 20, 15, 30, 30, 20]
# j = 8
# pred.predict(s, j)


### Sensors
# for sensor in ego.get_sensors():
#     print(sensor.name, end=", ")
#     # print(sensor.__dict__)
#     print(sensor.enabled)
#     # if sensor.name == "Lidar":
#     #     print(f"min_distance: {sensor.min_distance}")
#     #     print(f"max_distance: {sensor.max_distance}")
#     #     print(f"rays: {sensor.rays}")
#     #     print(f"rotations: {sensor.rotations}") # rotation frequency, Hz
#     #     print(f"measurements: {sensor.measurements}") # = j["measurements"]  # how many measurements each ray does per one rotation
#     #     print(f"fov: {sensor.fov}")
#     #     print(f"angle: {sensor.angle}")
#     #     print(f"compensated: {sensor.compensated}")

#     if sensor.name == "IMU":
#         # print(type(sensor))
#         print(sensor.__init__)
#         # for i in sensor:
#         #     print(i)

# print(sensor.name for sensor in sensors)



