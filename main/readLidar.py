import os
import numpy as np
import open3d as o3d
from accepts import accepts

PATH = os.path.abspath(__file__)
PATH = PATH[:PATH[:PATH.rfind("\\")].rfind("\\")]


class ReadLidar():
    def __init__(self, window=1.4, rays=20, filename=PATH+"/data/lidarUpdate.pcd"):
        """
        Get distance to objects by using a .pcd file.
        ### Params:
            * window: float, how many meters left/right of the center the algorithm looks for objects in front of the vehicle. The vehicle is around 2.6(?) m wide.
            * rays: int, how many rays the algorithm searches for an obect, higher value = longer reach, max 32(?)
            * filename: str, name of the .pcd file that keeps being reuploaded during simulation
        """
        self.mid = 180
        self.vehicleHeight = 0.
        self.ground = -2.0 # Represents the lidar z-coordinate that corresponds approximately to the ground
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
        ### NOTE: Not used, only used for visualizing
        Store only the points that are in front of the vehicle +/- the window size.\\
        Skips the first 720 points as some of them hits the EGO vehicle's bonnet and trunk
        """
        # self.inFront = np.array([x for (x, i) in zip(self.vectorList[:360*self.rays], self.indexes) if i % 360 >= self.mid-self.window and i % 360 <= self.mid+self.window])
        # self.inFront = np.array([x for (x, i) in zip(self.vectorList[:360*self.rays], self.indexes) if i % 360 >= self.mid and i % 360 <= self.mid])
        # print(len(self.inFront))
        self.inFront = np.array([(x, y, z) for (x, y, z) in self.vectorList[720:360*self.rays] if x > 0 and y > -self.window and y < self.window])
        self.inFront = np.array([(x, y, z) for (x, y, z) in self.vectorList[720:360*self.rays] if x > 2.86 and y > -self.window and y < self.window and z < self.vehicleHeight])

    def getDistanceToObstacle(self):
        """
        Gets the distance from the vehicle's bumper to the nearest obstacle in front of the vehicle.\\
        Ground at origo has coordiantes approximately (0, 0, -2.3).

        ### NOTE
        The lidar sees things from the roof of the EGO vehicle, and when looking at the distances, the lidar has
        readings from hitting the bonnet and trunk of its own vehicle (2.16 m and 2.33 m). The next thing it sees is the ground
        (4.33 m). If it has collided with a high wall, that same value is 2.87 m, which implies that the front of the
        bumper is 2.87 m away in a horizontal line from where the lidar is placed. This then has to be remembered
        when calculating the distance to object in front of the vehicle.\\
        NOTE The value self.ground and self.vehicleHeight might need some tuning.
        ### Returns:
            * the distance to the obstacle right in front of the vehicle that it might hit.

        TODO: See at other points as well that are not directly in front.
        """
        self.closestPointInFront = (102.87,0,-1) # (102.87,0,-1)
        # for x, y, z in self.inFront:
        #     # print(x, y, z)
        #     # if z > self.ground and z < self.vehicleHeight and np.sqrt(x**2+y**2) < np.sqrt(self.closestPointInFront[0]**2+self.closestPointInFront[1]**2):
        #         # print(x)
        #         self.closestPointInFront = (x, y, z)
        # # return self.closestPointInFront
        # return round(np.sqrt((self.closestPointInFront[0] - 2.87)**2+self.closestPointInFront[1]**2), 4)
        # self.closestPointInFront = 102.87
        for x, y, z in np.array([(x, y, z) for (x, y, z) in self.vectorList[720:360*self.rays] if x > 2.86 and y > -self.window and y < self.window and z > self.ground and z < self.vehicleHeight]):
            self.closestPointInFront = (x, y, z) if x < self.closestPointInFront[0] else self.closestPointInFront
        return round(self.closestPointInFront[0] - 2.87, 4)
    

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
            if x > 0 and y > -self.window and y < self.window and z > self.ground and z < self.vehicleHeight and (x, y, z) == self.closestPointInFront:
                # print(x, y, z)
                # print(x, round(x-2.87, 4))
                # print(ray, ray*360+self.mid-width, ray*360+self.mid+width)
                distances = self.vectorList[index-width: index+width+1]
                highestDiff = 0
                indexHighestDiff = 0
                for i, (x, y, z) in enumerate(distances[1:], 1):
                    distToPrevious = np.sqrt(distances[i-1][0]**2+distances[i-1][1]**2) - np.sqrt(distances[i][0]**2+distances[i][1]**2)
                    highestDiff, indexHighestDiff = (distToPrevious, i) if np.absolute(distToPrevious) > np.absolute(highestDiff) else (highestDiff, indexHighestDiff)
                return 1 if highestDiff >= 0 else -1
                # break
        else:
            return None

        highestDiff = 0
        indexHighestDiff = 0
        # print(0, distances[0])
        if distances != None:
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
        Propery to get the updated DTO. Reads a new point cloud to calculate a new DTO each time it is called.
        """
        self.readPCD()
        return self.getDistanceToObstacle()


if __name__ == "__main__":
    # lidar = ReadLidar(window=0.5, rays=20, filename=".\MasterThesis\data\\album\lidar8.pcd")
    lidar = ReadLidar(window=1.4, rays=35)
    # lidar = ReadLidar(window=1.4, rays=35, filename=".\MasterThesis\data\lidar\lidar20mSedan.pcd")
    lidar.readPCD()

    lidar.getPointsInFront()
    print(lidar.inFront)
    lidar.vizualizePointCloud(lidar.inFront)
    # closest = lidar.getDistanceToObstacle()
    # print(lidar.getDistanceToObstacle())#, np.sqrt(closest[0]**2+closest[1]**2))
    # print(lidar.closestPointInFront)
    # print(lidar.inFront)
    print(lidar.updatedDTO)
    evasive = lidar.getEvasiveAction(10)
    print(evasive)

        
