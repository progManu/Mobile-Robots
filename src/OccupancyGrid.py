import math

import src.Utilities as Utilities


class OccupancyGrid:
    def __init__(self, lidar, cell_dim=None, initial_position=None):
        if cell_dim is None:
            cell_dim = 1
        if initial_position is None:
            initial_position = [0, 0]
        self.initial_position = initial_position
        self.position = initial_position
        self.cell_dim = cell_dim
        self.lidar = lidar
        self.number_cells_margin = math.ceil((self.lidar.reach / self.cell_dim) / 2)
        self.grid = {}
        self.set_grid()

    def set_grid(self):
        for i in range(-self.number_cells_margin, self.number_cells_margin + 1):
            for j in range(-self.number_cells_margin, self.number_cells_margin + 1):
                x = self.initial_position[0] + i * self.cell_dim
                y = self.initial_position[1] + j * self.cell_dim
                self.grid[(x, y)] = 0

    def enlarge_grid_if_needed(self):
        robot_grid_position = (round(self.position[0] / self.cell_dim) * self.cell_dim,
                               round(self.position[1] / self.cell_dim) * self.cell_dim)
        for i in range(-self.number_cells_margin, self.number_cells_margin + 1):
            for j in range(-self.number_cells_margin, self.number_cells_margin + 1):
                x = robot_grid_position[0] + i * self.cell_dim
                y = robot_grid_position[1] + j * self.cell_dim
                if (x, y) not in self.grid:
                    self.grid[(x, y)] = 0

    def mark_cells_from_measurement(self, angle, measurement):
        lidar_segment_center = (self.position[0], self.position[1])
        lidar_segment_edge = (self.position[0] + measurement * math.cos(angle),
                              self.position[1] + measurement * math.sin(angle))
        # Here can be optimized by considering only the cells that are in the circle of radius measurement
        for cell_center in self.grid:
            # First check if the cell has already been marked as an obstacle
            if self.grid[cell_center] == -1:
                continue
            # Then check if the cell is an obstacle
            if Utilities.distance_point_point(cell_center, lidar_segment_edge) and measurement < self.lidar.reach:
                self.grid[cell_center] = -1
                continue
            # Finally, check if the cell is traversed by the segment
            distance_good = Utilities.distance_point_segment([lidar_segment_center, lidar_segment_edge], cell_center)
            # Actually this is a simplification, it considers the circle centered in the cell instead of the square
            if distance_good <= math.sqrt(2 * self.cell_dim ** 2) and self.grid[cell_center] == 0:
                self.grid[cell_center] = 1

    def update_grid(self, robot_current_position):
        self.position = robot_current_position
        self.enlarge_grid_if_needed()
        measurements = self.lidar.measure(robot_current_position)
        for key in measurements:
            measurement = measurements[key][0]
            if measurement is None:
                measurement = self.lidar.reach
            self.mark_cells_from_measurement(key, measurement)
