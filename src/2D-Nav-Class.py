import numpy as np

import scipy
from sympy import symbols, Eq, solve, diff, sin, cos
from scipy.integrate import odeint
from scipy.io import loadmat

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge
from matplotlib.lines import Line2D
from IPython.display import HTML
import matplotlib_inline.backend_inline
import matplotlib.gridspec as gridspec

import math

from random import seed
from random import random

import itertools

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import networkx as nx

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString



class Robot:

  matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
  matplotlib.rcParams.update({'font.size': 7})
  matplotlib.rcParams['animation.embed_limit'] = 100 # MB


  def __init__(self, robot_radius=1, wheels_distance=0.8, map_dimensions=[50,50], epsilon_collisions = 0.1,
               target_tollerance = 2, in_map_obstacle_vertexes_list=[], exp_title="NoName", plots_margins = 10):
    """
    Initialize the robot.

    Args:
      ## TO DO - Explain arguments
    """
    self.set_robot_radius(robot_radius)
    self.set_wheels_distance(wheels_distance)
    self.set_map_dimensions(map_dimensions)
    self.set_in_map_obstacles(in_map_obstacle_vertexes_list)
    self.set_epsilon_collisions( epsilon_collisions)
    self.set_target_tollerance(target_tollerance)
    self.title = exp_title
    self.plots_margins = plots_margins

  def set_exp_title(self, title: str):
    self.title = title

  def set_robot_radius(self, robot_radius):
    self.robot_radius = robot_radius

  def set_wheels_distance(self, wheels_distance):
    self.wheels_distance = wheels_distance

  def set_map_dimensions(self, map_dimensions):
    self.map_dimensions = map_dimensions
    self.map_x_len = self.map_dimensions[0]
    self.map_y_len = self.map_dimensions[1]
    self.map_vertexes = [(0,0), (0,self.map_y_len), (self.map_x_len,self.map_y_len), (self.map_x_len,0)]
    self.map_border_polygon = Polygon(self.map_vertexes)
    self.map_border_segments = self.polygon_to_segments(self.map_border_polygon)

  def set_epsilon_collisions(self, epsilon_collisions):
    self.epsilon_collisions = epsilon_collisions

  def set_target_tollerance(self, target_tollerance):
    self.target_tollerance = target_tollerance

  def set_in_map_obstacles(self, in_map_obstacle_vertexes_list):
    self.in_map_polygons = self.polygons_from_vertexes_list(in_map_obstacle_vertexes_list)
    self.in_map_obstacles_segments = self.polygons_to_segments(self.in_map_polygons)

  def set_initial_state(self, initial_state):
    self.initial_state = initial_state

  def set_final_position(self, final_position):
    self.final_position = final_position

  def set_cell_dimension(self, cell_x_width, cell_y_width):
    self.cell_x_width = cell_x_width
    self.cell_y_width = cell_y_width

  def set_time_sequence(self, final_time, sampling_time):
    self.sampling_time = sampling_time
    self.final_time = final_time
    self.simulation_steps = math.ceil(self.final_time/self.sampling_time)
    self.times = np.linspace(0, self.final_time, self.simulation_steps+1)



  def get_graph(self):
    return self.graph

  def get_polygons(self):
    return self.in_map_polygons



  def set_pd_controller(self, kp=0.1, kd=0):
    self.kp = kp
    self.kd = kd

  def simulation_setup(self, initial_state=[5,5,0], final_position=[45,45], final_time=10, sampling_time=0.01, cell_dimension=[5,5], cruise_velocity=3):
    self.initial_state = initial_state
    self.set_final_position(final_position)
    self.set_time_sequence(final_time, sampling_time)
    self.set_cell_dimension(cell_dimension[0], cell_dimension[1])
    self.cruise_velocity = cruise_velocity

  def simulate(self):
    self.simulator()

  def get_animation(self):
    return self.animation

  def save_animation(self, dpi=100):
    self.animation.save(filename=self.title+".mkv", writer="ffmpeg", dpi=dpi)

  def show_plots(self):
    self.plots().show()

  def save_plots(self, dpi=600):
    self.plots().savefig(self.title + ".png", bbox_inches='tight', transparent=False, dpi=dpi)



  def polygon_to_segments(self, polygon): #given a polygon, returns a list of all the segments that form its boundary
    points = []
    lines = []
    points = list(polygon.exterior.coords)
    points.remove(points[len(points)-1])
    for i in range(len(points)):
      p1 = points[i]
      if i != len(points) - 1:
        p2 = points[i+1]
      else:
        p2 = points[0]
      line = (p1,p2)
      lines.append(line)
    return lines

  def polygons_from_vertexes_list(self, vertexes_list):
    polygons = []
    for vertex in vertexes_list:
      polygons.append(Polygon(vertex))
    return polygons

  def polygons_to_segments(self, polygons):
    segments = []
    try:
      for i in range(len(polygons)):
        for segment in self.polygon_to_segments(polygons[i]):
          segments.append(segment)
      return segments
    except Exception as e:
      return segments

  def minimum_distance(self, obstacle, point):
    v = obstacle[0]
    w = obstacle[1]
    # Return minimum distance between line segment vw and point p
    v, w, p = np.array(v), np.array(w), np.array(point)
    l2 = np.sum((w - v)**2)  # i.e. |w-v|^2 -  avoid a sqrt
    if l2 == 0.0:
      return np.linalg.norm(p - v)   # v == w case
    # Consider the line extending the segment, parameterized as v + t (w - v).
    # We find projection of point p onto the line.
    # It falls where t = [(p-v) . (w-v)] / |w-v|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = max(0, min(1, np.dot(p - v, w - v) / l2))
    projection = v + t * (w - v)  # Projection falls on the segment
    return np.linalg.norm(p - projection)

  def check_obstacles(self, point):    # r is the ray of the "car" (sphere)
    for k, obstacle in enumerate(self.map_border_segments):
      # Calculate the distance between the line segment and the center of the sphere
      dist = self.minimum_distance(obstacle, [point[0], point[1]])
      if dist <= (self.robot_radius + self.epsilon_collisions):
        return False
    try:
      for k, obstacle in enumerate(self.in_map_obstacles_segments):
        # Calculate the distance between the line segment and the center of the sphere
        dist = self.minimum_distance(obstacle, [point[0], point[1]])
        if dist <= (self.robot_radius + self.epsilon_collisions):
          return False
    except Exception as e:
      return True
    # If the execution arrives here it means that car is not colliding with any obstacle
    return True


  def edge_is_not_valid(self, line): #check if the edge does not intersect any polygon nor it is too close to polygon vertices
    intersects = False
    try:
      for polygon in self.in_map_polygons:
        if intersects == False:
          if line.intersects(polygon):
            intersects = True
        for point in polygon.boundary.coords:
          p_point = Point(point)
          if intersects == False:
            if line.intersects(p_point.buffer(self.robot_radius)):
              intersects = True
    except Exception as e:
      intersects = False
    return intersects

  def set_graph_nodes(self):
    graph_nodes = []
    deleted_nodes = []
    p_graph_nodes = []
    for i in range(0, self.map_x_len+1, self.cell_x_width):
      for j in range(0, self.map_y_len+1, self.cell_y_width):
        p_graph_nodes.append(Point(i,j)) #generate all possible graph points
    for p_point in p_graph_nodes:
      contained = False
      p_circle = p_point.buffer(self.robot_radius)
      point = (p_point.x, p_point.y)
      sufficiently_far = self.check_obstacles(point)
      try:
        for polygon in self.in_map_polygons:
          if contained == False:
            if polygon.contains(p_point):
              contained = True
      except Exception as e:
        contained == False
      if contained == False and sufficiently_far == True: #a graph node must not be contained by a polygon nor be close to a polygon vertex
        graph_nodes.append(point)
      else:
        deleted_nodes.append(point)
    self.graph_nodes = graph_nodes

  def add_neighbor_if_possible(self, node, neighbor, neighbors):
    if neighbor in self.graph_nodes:
      p_line = LineString([node, neighbor])
      if not self.edge_is_not_valid(p_line): #add edge to neighborhood if it is valid
        neighbors.append(neighbor)

  def create_graph_from_nodes(self):
    graph = {}
    for node in self.graph_nodes:
      neighbors = []
      self.add_neighbor_if_possible(node, (node[0]+self.cell_x_width,node[1]), neighbors) #check edge E
      self.add_neighbor_if_possible(node, (node[0]-self.cell_x_width,node[1]), neighbors) #check edge W
      self.add_neighbor_if_possible(node, (node[0],node[1]+self.cell_y_width), neighbors) #check edge N
      self.add_neighbor_if_possible(node, (node[0],node[1]-self.cell_y_width), neighbors) #check edge S
      self.add_neighbor_if_possible(node, (node[0]+self.cell_x_width,node[1]-self.cell_y_width), neighbors) #check edge NW
      self.add_neighbor_if_possible(node, (node[0]+self.cell_x_width,node[1]+self.cell_y_width), neighbors) #check edge NE
      self.add_neighbor_if_possible(node, (node[0]-self.cell_x_width,node[1]-self.cell_y_width), neighbors) #check edge SW
      self.add_neighbor_if_possible(node, (node[0]-self.cell_x_width,node[1]+self.cell_y_width), neighbors) #check edge SE
      if neighbors != []:
        graph[node] = neighbors
    self.graph = graph # graph is a dictionary

  def get_nodes_and_edges(self):
    nodes = list(self.graph.keys())
    edges = []
    for node in nodes:
      for end_node in self.graph[node]:
        edges.append((node, end_node))

    return nodes, edges

  def find_path(self):
    nodes, edges = self.get_nodes_and_edges()
    graphG = nx.Graph()
    graphG.add_nodes_from(nodes)
    graphG.add_edges_from(edges)
    nearest_node_distance = 10**10
    initial_node = (0, 0)
    for key in self.graph:
      current_distance = math.dist(key, tuple(self.initial_state[:2]))
      if current_distance < nearest_node_distance:
        nearest_node_distance = current_distance
        initial_node = key
    shortest_path = nx.shortest_path(graphG, source=initial_node, target=tuple(self.final_position))
    self.nodes_path = shortest_path


  def plots(self):
    # Set the figure size
    plt.figure(figsize=(10, 9))
    # Set a grid of figures
    gs = gridspec.GridSpec(3, 2)
    # Create the subplots
    axs = [plt.subplot(gs[:2, 0]), plt.subplot(gs[2,:]), plt.subplot(gs[:2, 1])]
    # Set the title of the set of plots
    fig = plt.gcf()
    fig.suptitle(self.title, size=15, fontweight="bold")

    # Plot the movement of x_0
    min_x = 0-self.plots_margins
    max_x = self.map_x_len+self.plots_margins
    min_y = 0-self.plots_margins
    max_y = self.map_y_len+self.plots_margins
    axs[0].set_xlim([min_x, max_x])
    axs[0].set_ylim([min_y, max_y])
    axs[0].set_title(r'Position trajectory', fontweight = "bold")
    axs[0].plot(self.x_movement[:,0], self.x_movement[:,1], 'r')
    axs[0].set_xlabel(r'$x$ [m]')
    axs[0].set_ylabel(r'$y$ [m]')
    axs[0].set_aspect('equal')  # Set aspect ratio to 'equal' for square shape
    # Disegna ogni segmento
    for segment in self.map_border_segments:
      axs[0].plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')
    try:
      for segment in self.in_map_obstacles_segments:
        axs[0].plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')
    except Exception as e:
      do_nothing = True
    # Disegna target
    axs[0].plot(self.final_position[0], self.final_position[1], '*', markersize=15, color='green')  # '*' draw a star
    # Add final target node to plot
    final_target_node = self.target_node_movement[-1]
    axs[0].plot(final_target_node[0], final_target_node[1], '.', markersize=10, color='red')

    # Plot the movement of x_1
    axs[1].set_title(r'Velocities', fontweight = "bold")
    line1, = axs[1].plot(self.times[:len(self.times)-1], self.u_sequence[:,0], 'b', label=str(r'v_left'))
    line2, = axs[1].plot(self.times[:len(self.times)-1], self.u_sequence[:,1], 'r', label=str(r'v_right'))
    axs[1].set_xlabel(r'$t$ [s]')
    axs[1].set_ylabel(r'$v$ [m/s]')
    y_1_min_value = min(0, np.nanmin(self.u_sequence[:,0]),np.nanmin(self.u_sequence[:,1]))
    y_1_max_value = max(np.nanmax(self.u_sequence[:,0]),np.nanmax(self.u_sequence[:,1]))
    axs[1].set_ylim(y_1_min_value, y_1_max_value*1.1)
    axs[1].legend(handles=[line1, line2])
    axs[1].grid()


    # Plot graph and path
    axs[2].set_title(r'Graph and nodes path', fontweight = "bold")
    axs[2].set_aspect('equal')  # Set aspect ratio to 'equal' for square shape
    axs[2].set_xlim([min_x, max_x])
    axs[2].set_ylim([min_y, max_y])
    axs[2].set_xlabel(r'$x$ [m]')
    axs[2].set_ylabel(r'$y$ [m]')
    try:
      for segment in self.in_map_obstacles_segments:
        x1, y1 = [segment[0][0],segment[1][0]],[segment[0][1],segment[1][1]]
        axs[2].plot(x1, y1, 'k-')
    except Exception as e:
      do_nothing = True
    for key in self.graph:
      axs[2].plot(key[0],key[1], 'o', color=(0.6, 0.6, 1.0, 0.35)) #print the graph nodes
      for neighbor in self.graph[key]:
        x1, y1 = [key[0],neighbor[0]],[key[1],neighbor[1]]
        axs[2].plot(x1, y1, 'o-', color=(0.6, 0.6, 1.0, 0.35)) #print the edge

    for i in range(len(self.nodes_path)):
      if i != len(self.nodes_path) - 1:
        x1, y1 = [self.nodes_path[i][0],self.nodes_path[i+1][0]],[self.nodes_path[i][1],self.nodes_path[i+1][1]]
        plt.plot(x1, y1, 'o-', color=(0.1, 0.1, 1.0, 1)) #print the edge of the path

    # Disegna bordi arena
    for segment in self.map_border_segments:
      axs[2].plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')

    axs[2].plot(self.initial_state[0], self.initial_state[1], '.', markersize=12, color='red')
    axs[2].plot(self.final_position[0], self.final_position[1], '*', markersize=15, color='green')

    axs[2].set_xticks(range(0, self.map_x_len, 5))
    axs[2].set_yticks(range(0, self.map_y_len, 5))
    # Prevent plots from overlapping
    plt.tight_layout()
    # Display plots
    return plt


  def unwrap_angle(self, angle):
    two_pi = 2 * math.pi
    number_of_wraps = abs(angle)//two_pi
    if angle >= 0:
      unwrapped_angle = angle - (number_of_wraps * two_pi)
    else:
      unwrapped_angle = two_pi + angle + (number_of_wraps * two_pi)
    return unwrapped_angle

  def pd(self, target_heading, current_heading, previous_heading_error):
    two_pi = 2*math.pi
    unwrapped_target_heading = self.unwrap_angle(target_heading)
    unwrapped_current_heading = self.unwrap_angle(current_heading)
    heading_difference = 0
    proportional_error = 0
    if unwrapped_current_heading< unwrapped_target_heading:
      heading_difference = unwrapped_target_heading - unwrapped_current_heading
      if heading_difference>math.pi:
        proportional_error = heading_difference - two_pi
      else:
        proportional_error = heading_difference
    else:
      heading_difference = unwrapped_current_heading - unwrapped_target_heading
      if heading_difference>math.pi:
        proportional_error = -(heading_difference - two_pi)
      else:
        proportional_error = -heading_difference
    derivative = (proportional_error - previous_heading_error)/self.sampling_time
    proportional_term = self.kp * proportional_error
    derivative_term = self.kd * derivative
    delta = proportional_term + derivative_term
    return delta, proportional_error


  def differentialRobot_model(self, X, t, u, symbolic = False):
    # Set parameters
    r = self.robot_radius # car radius (the car is modeled as a sphere)
    d = self.wheels_distance  # distance between wheels
    # Change state value references for faster and clearer code
    x = X[0]
    y = X[1]
    angle = X[2] # Robot direction angle
    # Change inputs valuer references for faster and clearer code
    v_r = u[0] # Right wheel input velocity
    v_l = u[1] # Left wheel input velocity
    # Compute derivatives in local frame
    d_x_local = (v_r + v_l) / 2
    d_y_local = 0  # Assuming no motion in y-direction in local frame
    d_angle_local = (v_r - v_l) / d
    # Define rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    # Compute derivatives in global frame by rotating local frame derivatives
    local_frame = np.array([d_x_local, d_y_local, d_angle_local])
    global_frame = np.dot(R, local_frame)
    # Extract global frame derivatives
    d_x = global_frame[0]
    d_y = global_frame[1]
    d_angle = global_frame[2]
    # Return an array of derivatives
    return [d_x, d_y, d_angle]



  def average_target_node_distance(self):
    if self.target_node_movement is None or self.x_movement is None:
      raise ValueError("Target node movement or actual node movement data not provided.")

    distances = []
    for i in range(self.final_times_index): #target_node, actual_node in zip(self.target_node_movement, self.x_movement):
      target_position = np.array(self.target_node_movement[i])
      actual_position = np.array(self.x_movement[i][:2])
      distance = np.linalg.norm(target_position - actual_position)
      distances.append(distance)

    average_distance = np.mean(distances)
    print(average_distance)

  def get_fitness_value(self, kp, kd):
    self.set_pd_controller(kp=kp, kd=kd)
    self.simulate()
    return self.average_target_node_distance()



  def simulator(self):

    state_dim = len(self.initial_state)
    input_dim = 2

    # Init the matrix that will contain the state movement
    self.x_movement = np.full((self.simulation_steps + 1, state_dim), np.nan)
    self.x_movement[0] = self.initial_state # The first row of the matrix represents the initial conditions

    # Init an array that will contain the input sequence
    self.u_sequence = np.full((self.simulation_steps, input_dim), np.nan)

    self.set_graph_nodes()

    self.create_graph_from_nodes()

    initial_position = self.initial_state[:2]

    self.find_path()

    current_path_index = 0

    previous_heading_error = 0

    self.target_node_movement = []
    self.final_times_index = self.simulation_steps-1

    # Iterate over the number of steps
    for i in range(self.simulation_steps):

      # Get the current state
      self.state = self.x_movement[i]
      current_position = self.state[:2]
      current_heading = self.state[2]
      target_position = self.nodes_path[current_path_index]

      # print(i, current_heading)

      target_distance = np.linalg.norm(target_position - current_position)
      if target_distance < self.target_tollerance:
        if current_path_index == len(self.nodes_path)-1:
          print(f"Target reached in {i*self.sampling_time} seconds, simulation will be stopped!")
          self.final_times_index = i
          break
        current_path_index += 1
        target_position = self.nodes_path[current_path_index]  # If to close to the node, update it. This line is needed to prevent errors at initial time

      self.target_node_movement.append(target_position)

      target_heading = np.arctan2(target_position[1] - current_position[1], target_position[0] - current_position[0])

      delta_velocity, previous_heading_error = self.pd(target_heading=target_heading, current_heading=current_heading, previous_heading_error=previous_heading_error)

      target_distance = np.linalg.norm(target_position - current_position)
      # print(i/self.simulation_steps, delta_velocity, target_position, target_distance, previous_heading_error, target_heading, current_heading)

      # Compute the current input
      current_input = [self.cruise_velocity+delta_velocity, self.cruise_velocity-delta_velocity]
      self.u_sequence[i] = current_input

      # Define the time step associated to the current input
      time_interval = [self.times[i], self.times[i+1]]

      # Compute the state evolution keeping the final state, i.e. the state at time times[i+1]
      next_state = odeint(self.differentialRobot_model, self.state, time_interval, args=(current_input,))[1]
      # Don't update state when collisions are detected
      obstacle_collision = self.check_obstacles([self.state[0],self.state[1]])
      if obstacle_collision == False:
        print("A collision with an obstacle has been detected. The robot will be switched off for safety reasons!\n\n")
        self.final_times_index = i
        break

      self.x_movement[i+1] = next_state

    self.make_animation()


  def make_animation(self): # "obstacles" is a variable that contains

    times = self.times[:self.final_times_index]

    # Radius of the robot
    robot_radius = self.robot_radius

    # Setup Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis('equal') # Impose same scale for both axis

    # Define section for displayng time
    axtext = fig.add_axes([0.72,0.89,0.2,0.1])
    axtext.axis("off") # Turn the axis labels/spines/ticks off (not needed for visualising the clock)

    # Set the title of the set of plots
    # fig.suptitle('2D Navivation', size = 9)

    # Frame's dimensions
    min_x = 0-self.plots_margins
    max_x = self.map_x_len+self.plots_margins
    min_y = 0-self.plots_margins
    max_y = self.map_y_len+self.plots_margins
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    # Set some lables
    ax.set_xlabel(r'Position: $x$ [m]')
    ax.set_ylabel(r'Position: $y$ [m]')
    ax.set_title(self.title, size = 15, fontweight = "bold")

    # Disegna ogni segmento
    for segment in self.map_border_segments:
      ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')
    try:
      for segment in self.in_map_obstacles_segments:
        ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='black')
    except Exception as e:
      do_nothing = True

    # Initialize patch of the car
    car_center = (self.x_movement[0, 0], self.x_movement[0, 1])  # center of car
    heading_angle = self.x_movement[0, 2]
    car = Circle(car_center, robot_radius, color='magenta')  # Apply a circular patch (car)
    ax.add_patch(car)  # Add car patch to plot

    # Initialize line representing car's heading direction
    heading_line = Line2D([car_center[0], car_center[0] + robot_radius * np.cos(np.radians(heading_angle))],
                        [car_center[1], car_center[1] + robot_radius * np.sin(np.radians(heading_angle))],
                        linewidth=1.5, color='black')
    ax.add_line(heading_line)  # Add heading line to plot

    # Initialize scatter plot for target node
    target_node_marker = ax.scatter(self.target_node_movement[0][0], self.target_node_movement[0][1], color='red', marker='.')
    ax.plot(self.final_position[0], self.final_position[1], '*', markersize=15, color='green')  # '*' draw a star

    # Display experiment time
    time = axtext.text(0.5, 0.5, str(0))

    # Animate function
    def animate(i):
      # Update car representation:
      center = (self.x_movement[i, 0], self.x_movement[i, 1])
      heading_angle = np.rad2deg(self.x_movement[i, 2])
      car.center = center

      # Update heading line
      heading_line.set_xdata([center[0], center[0] + robot_radius * np.cos(np.radians(heading_angle))])
      heading_line.set_ydata([center[1], center[1] + robot_radius * np.sin(np.radians(heading_angle))])

      # Update position of target node marker
      target_node_marker.set_offsets(np.array([self.target_node_movement[i]]))

      # Update experiment time
      time.set_text('     Time: {0:.2f}'.format(times[i]) + str(r' $s$'))

      return car, heading_line, target_node_marker,

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(times),
                                  # Adjust the time interval between consecutive frames (1000 ms)
                                  interval = 1000 * (times[len(times)-1] - times[0]) / ((len(times) - 1)),
                                  blit=True)

    # Close the figure
    plt.close(fig)

    # Return the animation that has to be displayed
    self.animation = anim