import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import src.Utilities as Utilities
from shapely.geometry import LineString, Polygon


def get_code_meaning(code):
    if code == -1:
        return 'Obstacle'
    if code == 0:
        return 'No Information'
    if code == 1:
        return 'Frontier'
    if code == 2:
        return 'Free'
    return 'Unknown code'


class OccupancyGrid:
    def __init__(self, lidar, cell_dim, initial_state, nodes_distance_in_cells, minimum_distance_nodes_obstacles):
        ###############################################
        # nodes_distance_in_cells is expressed in cell_dim times
        ###############################################
        self.initial_position = initial_state[0:2]
        self.position = self.initial_position
        self.heading_angle = initial_state[2]
        self.cell_dim = cell_dim
        self.lidar = lidar

        self.number_cells_margin = math.ceil(self.lidar.reach / self.cell_dim)
        self.grid = {}
        self.set_grid()
        self.update_grid(initial_state)

        self.graph_nodes = {}
        self.graph = {}
        self.nodes_distance_in_cells = nodes_distance_in_cells
        self.minimum_distance_nodes_obstacles = minimum_distance_nodes_obstacles
        self.number_nodes_margin = math.floor(self.number_cells_margin / self.nodes_distance_in_cells)
        self.nodes_distance = self.nodes_distance_in_cells * self.cell_dim
        self.set_graph_nodes()
        self.update_graph(initial_state)

    #######################################################
    # NB: Each cell is represented by its down-left corner
    #######################################################

    def get_grid_position(self, position):
        return tuple([position[0] // self.cell_dim * self.cell_dim, position[1] // self.cell_dim * self.cell_dim])

    def set_grid(self):
        initial_cell = self.get_grid_position(self.initial_position)
        for i in range(-self.number_cells_margin, self.number_cells_margin + 1):
            for j in range(-self.number_cells_margin, self.number_cells_margin + 1):
                x = initial_cell[0] + i * self.cell_dim
                y = initial_cell[1] + j * self.cell_dim
                self.grid[(x, y)] = 0

    def set_graph_nodes(self):
        initial_cell = self.get_grid_position(self.initial_position)
        for i in range(-self.number_nodes_margin, self.number_nodes_margin + 1):
            for j in range(-self.number_nodes_margin, self.number_nodes_margin + 1):
                x = initial_cell[0] + i * self.nodes_distance
                y = initial_cell[1] + j * self.nodes_distance
                self.graph_nodes[(x, y)] = 0

    def enlarge_grid_if_needed(self):
        position_cell = self.get_grid_position(self.position)
        for i in range(-self.number_cells_margin, self.number_cells_margin + 1):
            for j in range(-self.number_cells_margin, self.number_cells_margin + 1):
                x = position_cell[0] + i * self.cell_dim
                y = position_cell[1] + j * self.cell_dim
                if (x, y) not in self.grid:
                    self.grid[(x, y)] = 0

    def enlarge_graph_nodes_if_needed(self):
        nearest_node = Utilities.get_nearest_node(self.graph_nodes, self.position)

        for i in range(-self.number_nodes_margin, self.number_nodes_margin + 1):
            for j in range(-self.number_nodes_margin, self.number_nodes_margin + 1):
                x = nearest_node[0] + i * self.nodes_distance
                y = nearest_node[1] + j * self.nodes_distance
                if (x, y) not in self.graph_nodes:
                    self.graph_nodes[(x, y)] = 0

    def mark_cells_from_measurement(self, lidar_intersection_point, lidar_measurement):
        segment_center = (self.position[0], self.position[1])
        segment_edge = lidar_intersection_point
        cells_diagonal = math.sqrt(2 * (self.cell_dim ** 2))

        lidar_segment_center_cell = self.get_grid_position(segment_center)
        if self.grid[lidar_segment_center_cell] == -1:
            raise ValueError('The robot is inside an obstacle, this should not happen')
        self.grid[lidar_segment_center_cell] = 2

        lidar_segment_edge_cell = self.get_grid_position(segment_edge)
        if lidar_measurement < self.lidar.reach:
            self.grid[lidar_segment_edge_cell] = -1
        if self.grid[lidar_segment_edge_cell] == 0:
            self.grid[lidar_segment_edge_cell] = 1

        delta_x_lidar = lidar_intersection_point[0] - segment_center[0]
        delta_y_lidar = lidar_intersection_point[1] - segment_center[1]

        if delta_x_lidar == 0 and delta_y_lidar == 0:
            return

        sampling_rate = 2 / self.cell_dim
        abs_delta_x_lidar = np.abs(delta_x_lidar)
        abs_delta_y_lidar = np.abs(delta_y_lidar)
        if abs_delta_x_lidar > abs_delta_y_lidar:
            number_of_steps = math.ceil(sampling_rate * abs_delta_x_lidar)
        else:
            number_of_steps = math.ceil(sampling_rate * abs_delta_y_lidar)

        x_step = delta_x_lidar / number_of_steps
        y_step = delta_y_lidar / number_of_steps
        lidar_segment_sampling = [(segment_center[0] + i * x_step, segment_center[1] + i * y_step)
                                  for i in range(1, number_of_steps - 1)]

        for point in lidar_segment_sampling:
            point_cell = self.get_grid_position(point)
            if point_cell == lidar_segment_center_cell or point_cell == lidar_segment_edge_cell:
                continue
            if self.grid[point_cell] != -1:
                self.grid[point_cell] = 2

    def get_cells_line_connecting_points(self, point_1, point_2):
        sampling_rate = 2 / self.cell_dim

        delta_x = point_2[0] - point_1[0]
        delta_y = point_2[1] - point_1[1]

        abs_delta_x = np.abs(delta_x)
        abs_delta_y = np.abs(delta_y)

        if abs_delta_x > abs_delta_y:
            number_of_steps = math.ceil(sampling_rate * abs_delta_x)
        else:
            number_of_steps = math.ceil(sampling_rate * abs_delta_y)

        x_step = delta_x / number_of_steps
        y_step = delta_y / number_of_steps
        segment_sampling = [(point_1[0] + i * x_step, point_1[1] + i * y_step)
                            for i in range(1, number_of_steps)]

        line_cells = []
        for point in segment_sampling:
            point_cell = self.get_grid_position(point)
            if point_cell not in line_cells:
                line_cells.append(point_cell)

        return line_cells

    def update_grid(self, robot_current_state):
        self.position = robot_current_state[0:2]
        self.heading_angle = robot_current_state[2]
        self.enlarge_grid_if_needed()
        lidar_results = self.lidar.measure(self.position, self.heading_angle)
        for lidar_angle, measurements in lidar_results.items():
            lidar_intersection_point = measurements[1]
            lidar_measurement = measurements[0]
            if lidar_intersection_point is None:
                lidar_intersection_point = (self.lidar.reach * math.cos(lidar_angle) + self.position[0],
                                            self.lidar.reach * math.sin(lidar_angle) + self.position[1])
                lidar_measurement = self.lidar.reach
            self.mark_cells_from_measurement(lidar_intersection_point, lidar_measurement)

    def check_node(self, node):

        number_of_cells = math.floor(self.minimum_distance_nodes_obstacles / self.cell_dim)

        central_cell_coordinates = self.get_grid_position(node)
        neighborhood = [
            (central_cell_coordinates[0] + i * self.nodes_distance,
             central_cell_coordinates[1] + j * self.nodes_distance)
            for i in range(-number_of_cells, number_of_cells + 1) for j in range(-number_of_cells, number_of_cells + 1)]
        node_value = 2
        for cell in neighborhood:
            if cell in self.graph_nodes:
                if self.graph_nodes[cell] == -1:
                    return -1
                elif self.graph_nodes[cell] == 1:
                    node_value = 1

        if node_value == 2:
            for cell in neighborhood:
                if cell in self.graph_nodes:
                    if self.graph_nodes[cell] == 0:
                        return 0

        return node_value

    def check_edge(self, node1, node2):

        n_cells_to_check = math.ceil(self.minimum_distance_nodes_obstacles / self.cell_dim)

        cells_line_connecting_nodes = self.get_cells_line_connecting_points(node1, node2)

        for cell in cells_line_connecting_nodes:
            for i in range(-n_cells_to_check, n_cells_to_check + 1):
                cell_to_check = cell[0] + i * self.cell_dim, cell[1]
                if (self.get_grid_position(cell_to_check)) in self.grid:
                    if self.grid[(cell[0] + i * self.cell_dim, cell[1])] != 2:
                        return False
                else:
                    return False

        return True

    def get_polygon_from_cell(self, cell):

        dl_corner = cell
        ul_corner = (dl_corner[0], dl_corner[1] + self.cell_dim)
        ur_corner = (dl_corner[0] + self.cell_dim, dl_corner[1] + self.cell_dim)
        dr_corner = (dl_corner[0] + self.cell_dim, dl_corner[1])

        polygon = Polygon([dl_corner, ul_corner, ur_corner, dr_corner])

        return polygon

    def get_node_neighbors(self, node):
        neighbors = []
        for neighbor in self.graph_nodes:
            if self.graph_nodes[neighbor] != 2 or neighbor == node:
                continue
            distance = Utilities.distance_point_point(node, neighbor)
            if distance <= self.nodes_distance * math.sqrt(2):
                neighbors.append(neighbor)
        return neighbors

    def get_reachable_neighbor_nodes(self, node):

        reachable_nodes = []
        for neighbor in self.get_node_neighbors(node):
            if self.check_edge(node, neighbor):
                reachable_nodes.append(neighbor)

        return reachable_nodes

    def update_graph(self, robot_current_state):

        self.position = robot_current_state[0:2]
        self.enlarge_graph_nodes_if_needed()

        # # nodes_to_check_arcs = []
        # nearest_node = Utilities.get_nearest_node(self.grid, self.position)
        # for i in range(-2 * self.number_nodes_margin, 2 * self.number_nodes_margin + 1):
        #     for j in range(-2 * self.number_nodes_margin, self.number_nodes_margin + 1):
        #         node = (nearest_node[0] + i * self.nodes_distance, nearest_node[1] + j * self.nodes_distance)
        #         if node in self.graph_nodes:
        #             node_value = self.check_node(node)
        #             self.graph_nodes[node] = node_value
        #             if node_value == 2:
        #                 nodes_to_check_arcs.append(node)

        # for node in nodes_to_check_arcs:
        for node in self.graph_nodes:
            reachable_neighbor_nodes = self.get_reachable_neighbor_nodes(node)
            self.graph[node] = reachable_neighbor_nodes

    def plot_grid(self):
        # Initialize the plot with a light gray background
        fig, ax = plt.subplots()
        ax.set_facecolor('lightgray')

        # Initialize the plot limits
        plot_min_x = 0
        plot_max_x = 0
        plot_min_y = 0
        plot_max_y = 0

        for key, value in self.grid.items():

            # Update the plot limits
            plot_min_x = min(plot_min_x, key[0])
            plot_max_x = max(plot_max_x, key[0])
            plot_min_y = min(plot_min_y, key[1])
            plot_max_y = max(plot_max_y, key[1])

            if value == -1:
                color = 'black'
            elif value == 0:
                color = 'lightgray'
            elif value == 1:
                color = 'orange'
            elif value == 2:
                color = 'white'
            else:
                raise ValueError('Unknown value in the grid')

            # Create a rectangle centered at 'key' with size 'self.dim x self.dim'
            rect = patches.Rectangle((key[0] - self.cell_dim / 2, key[1] - self.cell_dim / 2), self.cell_dim,
                                     self.cell_dim, linewidth=1, edgecolor=color, facecolor=color)
            ax.add_patch(rect)

        plt.xlim([plot_min_x - 10, plot_max_x + 10])
        plt.ylim([plot_min_y - 10, plot_max_y + 10])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


from src.LiDAR import LiDAR

if __name__ == "__main__":
    lidar = LiDAR()
    cell_dim = 1
    initial_state = (0, 0, 0)
    nodes_distance_in_cells = 1
    minimum_distance_nodes_obstacles = 1

    grid = OccupancyGrid(lidar, cell_dim, initial_state, nodes_distance_in_cells, minimum_distance_nodes_obstacles)

    grid.grid = {
        (0, 0): 2,
        (1, 0): 2,
        (2, 0): 2,
        (3, 0): 2,
        (4, 0): 2,
        (0, 1): 2,
        (1, 1): 2,
        (2, 1): 2,
        (3, 1): 2,
        (4, 1): 2,
        (0, 2): 2,
        (1, 2): 2,
        (2, 2): 2,
        (3, 2): 2,
        (4, 2): 2,
        (0, 3): 2,
        (1, 3): 2,
        (2, 3): 2,
        (3, 3): 2,
        (4, 3): 2,
        (0, 4): 2,
        (1, 4): 2,
        (2, 4): 2,
        (3, 4): 2,
        (4, 4): 2
    }

    grid.graph_nodes = {
        (1, 1): 2,
        (1, 2): 2,
        (1, 3): 2,
        (1, 4): 2,
        (2, 1): 2,
        (2, 2): 2,
        (2, 3): 2,
        (2, 4): 2,
        (3, 1): 2,
        (3, 2): 2,
        (3, 3): 2,
        (3, 4): 2,
        (4, 1): 2,
        (4, 2): 2,
        (4, 3): 2,
        (4, 4): 2

    }

    for node in grid.graph_nodes:
        reachable_neighbor_nodes = grid.get_reachable_neighbor_nodes(node)
        print(grid.get_node_neighbors(node))
        print(node, reachable_neighbor_nodes)
        print("\n\n")

    print(grid.get_cells_line_connecting_points((0.1, 0.3), (20.2, 2.1)))
