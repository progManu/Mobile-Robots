import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import src.Utilities as Utilities


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

    def set_grid(self):
        x_initial = self.initial_position[0] // self.cell_dim * self.cell_dim
        y_initial = self.initial_position[1] // self.cell_dim * self.cell_dim
        for i in range(-self.number_cells_margin, self.number_cells_margin + 1):
            for j in range(-self.number_cells_margin, self.number_cells_margin + 1):
                x = x_initial + i * self.cell_dim
                y = y_initial + j * self.cell_dim
                self.grid[(x, y)] = 0

    def enlarge_grid_if_needed(self):
        robot_grid_position = (self.position[0] // self.cell_dim * self.cell_dim,
                               self.position[1] // self.cell_dim * self.cell_dim)
        for i in range(-self.number_cells_margin, self.number_cells_margin + 1):
            for j in range(-self.number_cells_margin, self.number_cells_margin + 1):
                x = robot_grid_position[0] + i * self.cell_dim
                y = robot_grid_position[1] + j * self.cell_dim
                if (x, y) not in self.grid:
                    self.grid[(x, y)] = 0

    def mark_cells_from_measurement(self, lidar_intersection_point, lidar_measurement):
        lidar_segment_center = (self.position[0], self.position[1])
        lidar_segment_edge = lidar_intersection_point
        cells_diagonal = math.sqrt(2 * (self.cell_dim ** 2))
        cells_half_diagonal = cells_diagonal / 2

        for cell_down_left_corner in self.grid:
            # Consider only the cells that are within the reach of the lidar plus a margin of 2 cells to compensate
            # discretization errors
            if (Utilities.distance_point_point(cell_down_left_corner, self.position) >
                    self.lidar.reach + 2 * cells_diagonal):
                continue

            # First check if the cell has already been marked as an obstacle, in that case, skip it
            # This is useful to avoid overwriting obstacles
            if self.grid[cell_down_left_corner] == -1:
                continue

            cell_center = (cell_down_left_corner[0] + self.cell_dim / 2, cell_down_left_corner[1] + self.cell_dim / 2)

            # Then check if the cell is an obstacle
            if (Utilities.distance_point_point(cell_center, lidar_segment_edge) < cells_half_diagonal
                    and lidar_measurement < self.lidar.reach):
                self.grid[cell_down_left_corner] = -1
                continue

            # Finally, check if the cell is traversed by the segment
            distance_from_lidar_segment = Utilities.distance_point_segment(
                [lidar_segment_center, lidar_segment_edge], cell_center)
            # Actually this is a simplification, it considers the circle centered in the cell instead of the square
            if distance_from_lidar_segment < cells_half_diagonal:
                self.grid[cell_down_left_corner] = 1

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
                color = 'black'  # red for -1
            elif value == 1:
                color = 'white'  # green for 1
            else:
                color = 'yellow'  # yellow for 0

            # Create a rectangle centered at 'key' with size 'self.dim x self.dim'
            rect = patches.Rectangle((key[0] - self.cell_dim / 2, key[1] - self.cell_dim / 2), self.cell_dim,
                                     self.cell_dim, linewidth=1, edgecolor=color, facecolor=color)
            ax.add_patch(rect)

        plt.xlim([plot_min_x - 10, plot_max_x + 10])
        plt.ylim([plot_min_y - 10, plot_max_y + 10])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
