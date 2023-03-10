from environs import Env
import lgsvl
import matplotlib.pyplot as plt
import numpy as np

from ML.Model import Predicter

# TODO
# Seems like positions are wrong when loading files to the scenariorunner.
# Possibly make something to transform the positions.

# TODO
# Need a way to get DTO

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


def plotData(s, a, j, p, duration, interval):
    x = [i for i in range(len(s))]
    plt.plot(x, s, label="Speed m/s")
    plt.plot(x, a, "--", label="Acceleration m/s^2")
    plt.plot(x, j, ":", label="Jerk m/s^3")
    plt.plot(x, p, "-", label="Predicted crash (1 or 0)")
    plt.xlabel(f"Time, delta = {interval} s, duration = {duration} s")
    plt.legend()
    plt.title("Speed, acceleration, jerk and crash prediction")
    plt.grid()
    plt.show()


class P():
    def __init__(self, filename) -> None:
        self.model = Predicter()
        self.model.loadModel(filename)
        self.translate = {1: "You are going to crash!", 0: "Not crashing!"}

    def predict(self, speeds=[0,0,0,0,0,0], jerk=0, ttc=20, dto=20):
        """
        Used to predict: [TTC, DTO, Jerk, road, scenario, speed1, speed2, speed3, speed4, speed5, speed6]

        TODO
        Make sure the values are correct.
        Make it so prediction only takes from each second, if updateInterval is lower then 1.

        Attribute[TTC]	Attribute[DTO]	Attribute[Jerk]	reward	road	strategy	scenario	speed1	speed2	speed3	speed4	speed5	speed6
        100000.000000	6.434714	5.04	ttc	road2	random	rain_night	5.547	4.660	4.401	4.228	3.986	3.746
        
        """
        # x = np.array([20, 20, 1, 2, 1,1,2,3,4,5,6])
        x = [ttc, dto, jerk] + speeds
        xP, _ = self.model.preProcess(x)
        # print(f"x: {x}\t\tprocessed: {xP[0]}")
        prediction = self.model.predict(xP)[0]
        # return self.translate[prediction]
        return prediction
        


def runSimulation(map: str="bg", simDuration: float=10, updateInterval: float=1, plotting: bool=True):
    """
    Run simulation in LGSVL (OSSDC-SIM).

    Params:
        map: str, 'bg' for Borregas Avenue or 'sf' for San Francisco
        simDuration: float, time (seconds) for simulation duration
        updateInterval: float, time (seconds) between each data logging
        plotting: bool, plot speed, acceleration and jerk after the simulation
    """
    env = Env()
    sim = lgsvl.Simulator(env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host), env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))
    
    selectedMap = lgsvl.wise.DefaultAssets.map_borregasave
    if map != "bg":
        selectedMap = lgsvl.wise.DefaultAssets.map_sanfrancisco

    if sim.current_scene ==  selectedMap:
        sim.reset()
    else:
        sim.load(selectedMap)

    state = lgsvl.AgentState()
    spawns = sim.get_spawn()
    state.transform = spawns[1]
    # state.transform.position = position
    # state.transform.rotation = rotation
    # state.velocity = velocity
    # state.angular_velocity = angular_velocity

    ego = sim.add_agent(env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.ego_lincoln2017mkz_apollo5), lgsvl.AgentType.EGO, state)

    ego.on_collision(on_collision)
    ego.on_custom(on_custom)

    speeds = [0]
    acceleration = [0]
    jerk = [0]
    predictions = [0]*5

    pred = P("Classifier")

    while True:
        sim.run(updateInterval)

        speed = ego.state.speed
        speeds.append(round(speed, 3))
        acceleration.append((speeds[-1]-speeds[-2])/updateInterval)
        jerk.append((acceleration[-1]-acceleration[-2])/updateInterval)

        print(f"Speed: {round(speeds[-1], 2)} m/s")
        print(f"Acceleration: {round(acceleration[-1], 2)} m/s^2")
        print(f"Jerk: {round(jerk[-1], 2)} m/s^3")
        if len(speeds) > 5:
            predictions.append(pred.predict(speeds[-6:], np.average(jerk[-6:])))

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

        if len(speeds)*updateInterval > simDuration: break
    if plotting: plotData(speeds, acceleration, jerk, predictions, simDuration, updateInterval)
        


    

if __name__ == "__main__":
    file = "C:/MasterFiles/DeepScenario/deepscenario-dataset/greedy-strategy/reward-dto/road3-sunny_day-scenarios/0_scenario_8.deepscenario"
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

    s, a, j = runSimulation("sf", 30, 1)
    plotData(s,a,j)


