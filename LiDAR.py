import math
import numpy as np

class LiDAR:
    def __init__(self, angle_interval=[0, 2*math.pi], obstacles=[], n=10, reach=10):
        self.angle_interval = angle_interval
        self.obstacles = obstacles
        self.n = n
        self.reach = reach

    def line(p1, p2): #function for finding line coefficients given two points that define the line
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersection(L1, L2): #function for finding the intersection of the two lines
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False

    def measure_at_angle(self, angle, position): #function for measuring the distance between the robot position in a certain direction

        max_reach_point = (position[0] + (self.reach*np.cos(angle)), position[1] + (self.reach*np.sin(angle))) #finding the maximum point for computing intersections
        ray = ((position[0], position[1]),(max_reach_point[0], max_reach_point[1])) #generating the ray

        measured_distance = 2*self.reach #dummy value for beginning the procedure
        intersection_point = None

        for obstacle in self.obstacles: #repeat for each obstacle
        
          L1 = self.line(ray[0], ray[1]) #find coefficients of the ray
          L2 = self.line(obstacle[0], obstacle[1]) #find coefficients of the obstacle

          R = self.intersection(L1, L2) #find the intersection between the ray and the obstacle
          if not R: #if there is no intersection, the "out of range" value is sensed
            print("No intersection found with ", obstacle, ", getting maximum range")
          else: #there is an intersection
            distance = math.dist(position, R) #computing the distance between the actual position and the intersection
            if distance < measured_distance and distance <= self.reach: #if it is the current minimum distance and it is in reach
              if ((position[0]<=R[0]<=max_reach_point[0]) or (max_reach_point[0]<=R[0]<=position[0])) and ((position[1]<=R[1]<=max_reach_point[1]) or (max_reach_point[1]<=R[1]<=position[1])):
              #if it is in the direction of the ray
                if ((obstacle[0][0]<=R[0]<=obstacle[1][0]) or (obstacle[1][0]<=R[0]<=obstacle[0][0])) and ((obstacle[0][1]<=R[1]<=obstacle[1][1]) or (obstacle[1][1]<=R[1]<=obstacle[0][1])):
                #if the intersection is actually in the obstacle (not just in the direction of the obstacle)
                  measured_distance = distance #update the measured distance
                  intersection_point = R #the intersection point is the one computed before

        if measured_distance == 2*self.reach: #if no obstacle is detected, return None
          return None, None

        else: #else return the measured distance and the intersection point
          return measured_distance, intersection_point
        
    def measure(self, position):
        if self.n<=0:
            raise ValueError('The number of rays must be positive')
        
        angles = np.linspace(self.angle_interval[0], self.angle_interval[1], self.n +1)
        measurements = {}

        for angle in angles:
          measurement, intersection_point = self.measure_at_angle(angle, position)
          measurements[angle] = (measurement, intersection_point)

        return measurements
