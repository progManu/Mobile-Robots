import math


class ObstacleAvoidance:

    def __init__(self, lidar, safety_distance=9, critical_distance=1.1, k=1):
        self.lidar = lidar
        self.safety_distance = safety_distance
        self.k = k
        self.critical_distance = critical_distance

        if safety_distance > self.lidar.reach:
            print('Warning: safety distance is greater than the LiDAR reach')

        if self.lidar.n % 2 == 0:
            raise ValueError('LiDAR number of rays must be odd')

    def select_direction(self, robot_position):  # Returns the preferred side: 'Left' or 'Right'

        measurements = self.lidar.measure(robot_position)  # Scan the environment
        central_index = math.floor(len(measurements) / 2)  # Find central measurement

        partial_sums = {'right': 0, 'left': 0}

        list_measurements = []
        for key in measurements:  # Transform the dictionary into a list
            list_measurements.append((key, measurements[key][0], measurements[key][1]))

        for i in range(central_index):
            distance_measured = list_measurements[i][1]
            point = list_measurements[i][2]

            if point is not None:  # Sum the sensed distance for each ray in the first interval
                partial_sums['right'] += distance_measured
            else:
                partial_sums['right'] += self.lidar.reach

        for i in range(central_index, len(measurements)):
            distance_measured = list_measurements[i][1]
            point = list_measurements[i][2]

            if point is not None:  # Sum the sensed distance for each ray in the second interval
                partial_sums['left'] += distance_measured
            else:
                partial_sums['left'] += self.lidar.reach

        if partial_sums['right'] > partial_sums['left']:  # Return the largest sum
            return 'Right'
        else:
            return 'Left'

    def check_close_obstacles(self,
                              robot_position):  # Returns True if there are obstacles closer than the security distance
        measurements = self.lidar.measure(robot_position)  # Scan the environment

        list_dict = []
        for key in measurements:
            measurement = measurements[key][0]
            if measurement is None: measurement = self.lidar.reach
            list_dict.append([key, measurement])

        minimum_distance = min(list_dict, key=lambda x: x[1])[1]
        if minimum_distance < self.safety_distance:
            return True, minimum_distance
        else:
            return False, minimum_distance

    def compute_contribution(self, distance, robot_position):
        direction = self.select_direction(robot_position)
        contribution = self.k * 1 / distance

        if direction == 'Left':
            return -contribution, +contribution
        else:
            return +contribution, -contribution

    def distance_is_critical(self, robot_position):
        measurements = self.lidar.measure(robot_position)  #scan the environment

        list_dict = []
        for key in measurements:
            measurement = measurements[key][0]
            if measurement is None: measurement = self.lidar.reach
            list_dict.append([key, measurement])

        minimum_distance = min(list_dict, key=lambda x: x[1])[1]
        if minimum_distance < self.critical_distance:
            return True
        else:
            return False
