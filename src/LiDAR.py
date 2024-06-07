import math
import numpy as np
from shapely.geometry import LineString
import concurrent.futures
import matplotlib.pyplot as plt

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

    def measure(self, position, heading_angle):  # Repeat the measure at different angles
        # Added heading_angle to the measure function to allow the LiDAR to rotate with the robot
        if self.n <= 0:
            raise ValueError('The number of rays must be positive')

        angles = np.linspace(heading_angle + self.angle_interval[0], heading_angle + self.angle_interval[1], self.n + 1)

        measurements = {}

        for angle in angles:
            measurement, intersection_point = self.measure_at_angle(angle, position)
            measurements[angle] = (measurement, intersection_point)

        return measurements
    
    def plot_robot_and_rays(self, pos, measure):
        fig, ax = plt.subplots()

        coord = np.empty([2, len(measure)])

        for i, angle in enumerate(measure.keys()):
            dist, collision_pos = measure[angle]
            if dist is None:
                coord[0, i] = pos[0] + self.reach*np.cos(angle)
                coord[1, i] = pos[1] + self.reach*np.sin(angle)
            else:
                coord[0, i] = pos[0] + dist*np.cos(angle)
                coord[1, i] = pos[1] + dist*np.sin(angle)
            
            ax.plot([pos[0], coord[0, i]], [pos[1], coord[1, i]])
        
        print(coord)
        
        ax.plot(coord[0, :], coord[1, :], 'o')

        return fig
        



    # Multithreaded version of the measure function, it seems to be slower than the sequential version...
    # def measure(self, position, heading_angle):
    #    if self.n <= 0:
    #        raise ValueError('The number of rays must be positive')

    #    angles = np.linspace(self.angle_interval[0]+heading_angle, self.angle_interval[1]+heading_angle, self.n + 1)

    #    measurements = {}

    #    # Function to measure at a specific angle
    #    def measure_at_angle(angle):
    #        return self.measure_at_angle(angle, position)

    #    # Perform lidar measurements in parallel
    #    with concurrent.futures.ThreadPoolExecutor() as executor:
    #        futures = {executor.submit(measure_at_angle, angle): angle for angle in angles}
    #        for future in concurrent.futures.as_completed(futures):
    #            angle = futures[future]
    #            try:
    #                measurement, intersection_point = future.result()
    #                measurements[angle] = (measurement, intersection_point)
    #            except Exception as exc:
    #                print(f"Measurement at angle {angle} generated an exception: {exc}")

    #    return measurements
