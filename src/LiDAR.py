import math
import numpy as np
from shapely.geometry import LineString


class LiDAR:
    def __init__(self, angle_interval=None, obstacles=None, n=9, reach=10):
        if obstacles is None:
            obstacles = []
        if angle_interval is None:
            angle_interval = [0, 2 * math.pi]
        self.angle_interval = angle_interval
        self.obstacles = obstacles
        self.n = n
        self.reach = reach

    def measure_at_angle(self, angle, position):
        # Function for measuring the distance between the robot position in a certain direction

        max_reach_point = (position[0] + (self.reach * np.cos(angle)), position[1] + (
                    self.reach * np.sin(angle)))  # Finding the maximum point for computing intersections
        ray = ((position[0], position[1]), (max_reach_point[0], max_reach_point[1]))  # Generating the ray
        s_ray = LineString(
            [(position[0], position[1]), (max_reach_point[0], max_reach_point[1])])  # Converting the ray to a segment

        measured_distance = 2 * self.reach  # Dummy value for beginning the procedure
        intersection_point = None

        for obstacle in self.obstacles:  # Repeat for each obstacle

            s_obstacle = LineString([obstacle[0], obstacle[1]])

            if s_ray.intersects(s_obstacle):
                s_intersection_point = s_ray.intersection(s_obstacle)
                ray_until_obstacle = LineString(
                    [(position[0], position[1]), (s_intersection_point.x, s_intersection_point.y)])
                distance = ray_until_obstacle.length

                if distance < measured_distance and distance < self.reach:
                    measured_distance = distance
                    intersection_point = (s_intersection_point.x, s_intersection_point.y)

        if measured_distance == 2 * self.reach:  # If no obstacle is detected, return None
            return None, None

        else:  # Else return the measured distance and the intersection point
            return measured_distance, intersection_point

    def measure(self, position):  # Repeat the measure at different angles
        if self.n <= 0:
            raise ValueError('The number of rays must be positive')

        angles = np.linspace(self.angle_interval[0], self.angle_interval[1], self.n + 1)
        measurements = {}

        for angle in angles:
            measurement, intersection_point = self.measure_at_angle(angle, position)
            measurements[angle] = (measurement, intersection_point)

        return measurements
