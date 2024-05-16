import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import src.Utilities as Utilities


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
    def __init__(self, lidar, cell_dim, initial_state):
        self.initial_position = initial_state[0:2]
        self.position = self.initial_position
        self.heading_angle = initial_state[2]
        self.cell_dim = cell_dim
        self.lidar = lidar
        self.number_cells_margin = math.ceil(self.lidar.reach / self.cell_dim)
        self.grid = {}
        self.set_grid()
        self.update_grid(initial_state)

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

    def enlarge_grid_if_needed(self):
        position_cell = self.get_grid_position(self.position)
        for i in range(-self.number_cells_margin, self.number_cells_margin + 1):
            for j in range(-self.number_cells_margin, self.number_cells_margin + 1):
                x = position_cell[0] + i * self.cell_dim
                y = position_cell[1] + j * self.cell_dim
                if (x, y) not in self.grid:
                    self.grid[(x, y)] = 0

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

    # # This works poorly, it is better to use the method above
    # def mark_cells_from_measurement(self, lidar_intersection_point, lidar_measurement):
    #     lidar_segment_center = (self.position[0], self.position[1])
    #     lidar_segment_edge = lidar_intersection_point
    #     cells_diagonal = math.sqrt(2 * (self.cell_dim ** 2))
    #     cells_half_diagonal = cells_diagonal / 2
    #
    #     for cell_down_left_corner in self.grid:
    #         # Consider only the cells that are within the reach of the lidar plus a margin of 2 cells to compensate
    #         # discretization errors
    #         if (Utilities.distance_point_point(cell_down_left_corner, self.position) >
    #                 self.lidar.reach + 2 * cells_diagonal):
    #             continue
    #
    #         # First check if the cell has already been marked as an obstacle, in that case, skip it
    #         # This is useful to avoid overwriting obstacles
    #         if self.grid[cell_down_left_corner] == -1:
    #             continue
    #
    #         cell_center = (cell_down_left_corner[0] + self.cell_dim / 2, cell_down_left_corner[1] + self.cell_dim / 2)
    #         distance_from_lidar_edge = Utilities.distance_point_point(cell_center, lidar_segment_edge)
    #         # Then check if the cell is an obstacle
    #         if distance_from_lidar_edge < cells_half_diagonal and lidar_measurement < self.lidar.reach:
    #             self.grid[cell_down_left_corner] = -1
    #             continue
    #
    #         # Check if the cell is traversed by the lidar segment
    #         # Actually this is a simplification, it considers the circle centered in the cell instead of the square
    #         distance_from_lidar_segment = Utilities.distance_point_segment(
    #             [lidar_segment_center, lidar_segment_edge], cell_center)
    #         if distance_from_lidar_segment < cells_half_diagonal < distance_from_lidar_edge:
    #             self.grid[cell_down_left_corner] = 2
    #             continue
    #
    #         # Finally, check if the cell is a frontier point
    #         if distance_from_lidar_edge <= cells_half_diagonal and self.grid[cell_down_left_corner] == 0:
    #             self.grid[cell_down_left_corner] = 1
    #             continue

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
