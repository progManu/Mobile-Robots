import math
from shapely.geometry import LineString

class ObstacleAvoidance:

    def __init__(self, lidar, safety_distance):
        self.lidar = lidar
        self.safety_distance = safety_distance

        if self.lidar.n % 2 == 0:
            raise ValueError('LiDAR number of rays must be odd')

    
    def select_direction(self, robot_position): #returns the preferred side: 'Left' or 'Right' 

        measurements = self.lidar.measure(robot_position) #scan the environment
        central_index = math.floor(len(measurements) / 2) #find central measurement

        partial_sums = {'right': 0, 'left': 0}

        list_measurements = []
        for key in measurements: #transform the dictionary into a list
            list_measurements.append((key, measurements[key][0], measurements[key][1]))
        
        for i in range(central_index):
            distance_measured = list_measurements[i][1]
            point = list_measurements[i][2]

            if point != None: #sum the sensed distance for each ray in the first interval
                partial_sums['right'] += distance_measured
            else:
                partial_sums['right'] += self.lidar.reach

        for i in range(central_index, len(measurements)):
            distance_measured = list_measurements[i][1]
            point = list_measurements[i][2]

            if point != None: #sum the sensed distance for each ray in the second interval
                partial_sums['left'] += distance_measured
            else:
                partial_sums['left'] += self.lidar.reach

        if partial_sums['right'] > partial_sums['left']: #return the largest sum 
            return 'Right'
        else:
            return 'Left'
        
    def check_for_close_obstacles(self, robot_position): #returns True if there are obstacles closer than the security distance
        measurements = self.lidar.measure(robot_position) #scan the environment

        for key in measurements:
            if measurements[key][0] < self.safety_distance:
                return True
            
        return False