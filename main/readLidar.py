import numpy as np
import open3d as o3d
from accepts import accepts

import time


class ReadLidar():
    def __init__(self, window=0., rays=20, filename="../data/lidar.pcd"):
        """
        Get distance to objects by using a .pcd file.
        ### Params:
            * window: float, how many meters left/right of the center the algorithm looks for objects in front of the vehicle.
            * rays: int, how many rays the algorithm searches for an obect, more = longer reach, max 32(?)
            * filename: str, name of the .pcd file that keeps being reuploaded during simulation
        """
        self.mid = 180
        self.vehicleHeight = 0.2
        self.window = window if window > 0 else 0.1
        self.rays = rays if rays < 32 else 32
        self.indexes = np.arange(360*rays)
        self.closestPointInFront = None

        self.vectorList = None
        self.inFront = None
        self.filename = filename

    
    def readPCD(self):
        """
        Read and updates the point cloud object from a .pcd file and stores the point vectors.
        """
        try:
            pcd = o3d.io.read_point_cloud(self.filename)
            self.vectorList = np.asarray(pcd.points)
        except Exception as e:
            print(f"File is not found, error: {e}")


    # def loadPointCloud(self):
    #     """
    #     Loads the point vectors into a pointCloud object.
    #     """
    #     if isinstance(self.vectorList, np.ndarray):
    #         self.pointCloud.points = o3d.utility.Vector3dVector(self.vectorList)
    #     else:
    #         print("Need to load 'readPCD' beforehand!")

    
    @staticmethod
    @accepts(np.ndarray)
    def vizualizePointCloud(vectorPointList: np.ndarray=np.array([[-1,0,-1],[0,0,0],[1,0,1]])):
        """
        Shows the whole point cloud by sending in a list of vector coordinates.\\
        NOTE right hand system: x, y, z (front|behind, left|right, up|down).
        ### Params:
            * vectorPointList: np.ndarray, list of vector coordinates
        """
        pointCloud = o3d.geometry.PointCloud()
        pointCloud.points = o3d.utility.Vector3dVector(vectorPointList)
        # l = [0,0,0], [0,0,-2.31], [1,0,-2.31], [2,0,-2.31], [3,0,-2.31], [4,0,-2.31], [5,0,-2.31], [6,0,-2.31], [7,0,-2.31], [8,0,-2.31], [9,0,-2.31], [10,0,-2.31], [11,0,-2.31], [12,0,-2.31], [13,0,-2.31], [14,0,-2.31], [15,0,-2.31]
        # pointCloud.points = o3d.utility.Vector3dVector(np.vstack([l, vectorPointList]))
        o3d.visualization.draw_geometries([pointCloud])


    def getPointsInFront(self):
        """
        Store only the points that are in front of the vehicle +/- the window size.
        """
        # self.inFront = np.array([x for (x, i) in zip(self.vectorList[:360*self.rays], self.indexes) if i % 360 >= self.mid-self.window and i % 360 <= self.mid+self.window])
        # self.inFront = np.array([x for (x, i) in zip(self.vectorList[:360*self.rays], self.indexes) if i % 360 >= self.mid and i % 360 <= self.mid])
        # print(len(self.inFront))
        self.inFront = np.array([(x, y, z) for (x, y, z) in self.vectorList[720:360*self.rays] if x > 0 and y > -self.window and y < self.window])


    def getDistanceToObstacle(self):
        """
        Gets the distance from the vehicle's bumper to the nearest obstacle in front of the vehicle.\\
        Ground at origo has coordiantes approximately (0, 0, -2.3).

        ### NOTE
        The lidar sees things from the roof of the EGO vehicle, and when looking at the distances, the lidar has
        readings from hitting the bonnet of its own vehicle (2.16 m and 2.33 m). The next thing it sees is the ground
        (4.33 m). If it has collided with a high wall, that same value is 2.87 m, which implies that the front of the
        bumper is 2.87 m away in a horizontal line from where the lidar is placed. This then has to be remembered
        when calculating the distance to object in front of the vehicle.\\
        NOTE The value -2.1 might need some tuning.
        ### Returns:
            * the distance to the obstacle right in front of the vehicle.

        TODO: See at other points as well that are not directly in front.
        """
        self.closestPointInFront = (100,0,-1)
        for x, y, z in self.inFront:
            # print(x, y, z)
            if z > -2.1 and z < self.vehicleHeight and np.sqrt(x**2+y**2) < np.sqrt(self.closestPointInFront[0]**2+self.closestPointInFront[1]**2):
                # print(x)
                self.closestPointInFront = (x, y, z)
        # return self.closestPointInFront
        return round(np.sqrt((self.closestPointInFront[0] - 2.87)**2+self.closestPointInFront[1]**2), 4)
        return 100
    

    def getEvasiveAction(self, width=10):
        """
        Find out which way gives the best chance to not collide with an obstacle.\\
        Makes no difference that the lidar is not placed at the bumper while calculating the distances.

        ### TODO
            * Check if the suggested direction has room for the vehicle (might have to check other rays that can see further)
            * Check if it should break in stead
            * Consider position of point (index) when deciding, i.e. if -10 has distance 5.3, and 5 has 5.1, it should go towards 5 because it demands less steering
        ### Params:
            * width: int, how many points left or right of the obstacle the method should look for an opening
        ### Returns:
            * int: -1 or 1, suggests turning right if -1 or left if 1
        """
        # directlyInFront = np.array([x for (x, i) in zip(self.vectorList[:360*self.rays], self.indexes) if i % 360 >= self.mid and i % 360 <= self.mid])
        distances = None
        for index, (x, y, z) in enumerate(np.array([point for point in self.vectorList[720:360*self.rays]]), 720):
            if x > 0 and y > -self.window and y < self.window and z > -2.1 and z < self.vehicleHeight and (x, y, z) == self.closestPointInFront:
                # print(x, y, z)
                # if z > -2.1:
                # print("hei")
                # print(x, round(x-2.87, 4))
                # print(ray, ray*360+self.mid-width, ray*360+self.mid+width)
                # if distances is None:
                distances = self.vectorList[index-width: index+width+1]
                break

        highestDiff = 0
        indexHighestDiff = 0
        # print(0, distances[0])
        for i, (x, y, z) in enumerate(distances[1:], 1):
            # print(i, x, y, z, end=": \n")
            distToPrevious = np.sqrt(distances[i-1][0]**2+distances[i-1][1]**2) - np.sqrt(distances[i][0]**2+distances[i][1]**2)
            # print(f"DistToPrevious: {round(distToPrevious, 3)}")
            highestDiff, indexHighestDiff = (distToPrevious, i) if np.absolute(distToPrevious) > np.absolute(highestDiff) else (highestDiff, indexHighestDiff)
        # print(f"\t\tindex: {indexHighestDiff}, highestDiff: {highestDiff}, {1 if highestDiff >= 0 else -1}")
        return 1 if highestDiff >= 0 else -1
        return distances


    @property
    def updatedDTO(self):
        """
        Propery to get the updated DTO. Updates the DTO each time it is called.
        """
        self.readPCD()
        self.getPointsInFront()
        return self.getDistanceToObstacle()


if __name__ == "__main__":
    # lidar = ReadLidar(window=0.5, rays=20, filename=".\MasterThesis\data\\album\lidar8.pcd")
    lidar = ReadLidar(window=1, rays=35, filename=".\MasterThesis\data\lidar\lidar10mSedan.pcd")
    # lidar = ReadLidar(window=1, rays=35, filename=".\MasterThesis\data\lidar\lidarNoCars.pcd")
    lidar.readPCD()

    lidar.getPointsInFront()
    # print(lidar.inFront)
    # lidar.vizualizePointCloud(lidar.vectorList)
    closest = lidar.getDistanceToObstacle()
    print(lidar.getDistanceToObstacle())#, np.sqrt(closest[0]**2+closest[1]**2))
    # print(lidar.inFront)
    # print(lidar.updatedDTO)
    evasive = lidar.getEvasiveAction(10)
    print(evasive)

    # l= lidar.test()
    # print(l)
    # lidar.vizualizePointCloud(np.vstack([lidar.inFront, lidar.vectorList[:360*1]]))
    # lidar.vizualizePointCloud(np.vstack([lidar.inFront, evasive]))

    # for i in range(5,10):
    #     lidar = ReadLidar(window=0, rays=35, filename=f".\MasterThesis\data\\album\lidar{i}.pcd")
    #     lidar.readPCD()
    #     lidar.getPointsInFront()
    #     # print(lidar.updatedDTO)
    #     lidar.vizualizePointCloud(np.vstack([lidar.inFront, lidar.vectorList[:360*1]]))

        
