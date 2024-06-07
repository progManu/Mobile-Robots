from enum import IntEnum
from src.LiDAR import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class CellInfo(IntEnum):
    OBSTACLE = -1
    NO_INFO = 0,
    FRONTIER = 1,
    FREE = 2

class NewOccupancy:
    def __init__(self, pos, cell_dim=2, lidar_reach=10):
        if cell_dim % 2 == 0:
            self.cell_dim = cell_dim
        else:
            raise ValueError("Cell dimension must be even")
        
        self.grid = self.setup_grid(pos=pos)
        self.graph = nx.Graph()
        self.lidar_reach = lidar_reach
    
    def get_grid_position(self, pos: tuple) -> tuple:
        x, y = pos

        x = float(x)
        y = float(y)

        return (int(x // self.cell_dim)*self.cell_dim, int(y // self.cell_dim)*self.cell_dim)
    
    def setup_grid(self, pos: tuple, initial_grid_size=100) -> dict:
        grid = {}
        limit = int(initial_grid_size // 2)
        x, y = self.get_grid_position(pos)
        for j in range(-limit, limit + 1):
            for i in range(-limit, limit + 1):
                grid[(x + i*self.cell_dim, y + j*self.cell_dim)] = CellInfo.NO_INFO
        
        return grid
    
    def enlarge_grid_if_needed(self, pos):
        center_x, center_y = self.get_grid_position(pos)
        grid_resolution = self.cell_dim
        radius = 2*self.lidar_reach - grid_resolution
         
        x = np.arange(center_x - radius, center_x + radius + grid_resolution + 1, grid_resolution)
        y = np.arange(center_y - radius, center_y + radius + grid_resolution + 1, grid_resolution)
        xx, yy = np.meshgrid(x, y)

        # Filter points to be inside the circle
        distance = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        mask = distance < radius

        for x, y in zip(xx[mask], yy[mask]):
            pos = self.get_grid_position((x, y))
            if pos not in self.grid.keys():
                self.grid[pos] = CellInfo.NO_INFO

    
    
    def dda(self, x1, y1, x2, y2):
        eps = 0.00001

        start_pos = np.array([x1, y1])
        end_pos = np.array([x2, y2])

        start_pos_on_grid = np.array(list(self.get_grid_position((x1, y1))))
        end_pos_on_grid = np.array(list(self.get_grid_position((x2, y2))))

        ray_vector = end_pos - start_pos

        if abs(ray_vector[0]) < eps or abs(ray_vector[1]) < eps:
            return None
        
        basic_step = np.array([
                np.abs(self.cell_dim)*np.sqrt(1 + ((ray_vector[1]/ray_vector[0])**2)),
                np.abs(self.cell_dim)*np.sqrt(1 + ((ray_vector[0]/ray_vector[1])**2))
            ])

        ray_length_1D = np.zeros(2)

        step = [0, 0]

        steps = {}
        steps['x'] = []
        steps['y'] = []

        if ray_vector[0] < 0:
            step[0] = -self.cell_dim
            ray_length_1D[0] = (start_pos[0] - start_pos_on_grid[0])*basic_step[0]
        else:
            step[0] = self.cell_dim
            ray_length_1D[0] = ((start_pos_on_grid[0] + self.cell_dim) - start_pos[0])*basic_step[0]
        
        if ray_vector[1] < 0:
            step[1] = -self.cell_dim
            ray_length_1D[1] = (start_pos[1] - start_pos_on_grid[1])*basic_step[1]
        else:
            step[1] = self.cell_dim
            ray_length_1D[1] = ((start_pos_on_grid[1] + self.cell_dim) - start_pos[1])*basic_step[1]

        cells_to_move  = np.abs(end_pos_on_grid - start_pos_on_grid)/self.cell_dim

        max_idx = np.argmax(cells_to_move)

        while cells_to_move[max_idx] > 0:
            if ray_length_1D[0] < ray_length_1D[1]:
                start_pos_on_grid[0] += step[0]
                ray_length_1D[0] += basic_step[0]
                steps['x'].append(int(step[0]/self.cell_dim))
                steps['y'].append(0)

                if max_idx == 0:
                    cells_to_move[max_idx] -= 1
            else:
                start_pos_on_grid[1] += step[1]
                ray_length_1D[1] += basic_step[1]
                steps['x'].append(0)
                steps['y'].append(int(step[1]/self.cell_dim))

                if max_idx == 1:
                    cells_to_move[max_idx] -= 1
        return steps
            

    def mark_cells_in_ray(self, pos, angle, measurement):
        end_pos = np.empty(2)
        free_end_pos = False

        if measurement[0] is None:
            end_pos = pos + self.lidar_reach*np.array([np.cos(angle), np.sin(angle)])
            free_end_pos = True
        else:
            end_pos = np.array(measurement[1])
        
        start_pos_on_grid = np.array(self.get_grid_position(tuple(pos)))

        end_pos_on_grid = np.array(self.get_grid_position(tuple(end_pos)))

        steps = self.dda(pos[0], pos[1], end_pos[0], end_pos[1])

        if steps is None:
            return

        x_steps = steps['x']
        y_steps = steps['y']

        for x_step, y_step in zip(x_steps, y_steps):
            if self.grid[tuple(start_pos_on_grid)] == CellInfo.NO_INFO:
                self.grid[tuple(start_pos_on_grid)] = CellInfo.FREE
            start_pos_on_grid += np.array([x_step, y_step])*self.cell_dim
        
        if not free_end_pos: # due to the fact if the ray doesn't hit anything the cell is already set to FREE
            self.grid[tuple(end_pos_on_grid)] = CellInfo.OBSTACLE
    
    def plot_radius_and_grid(self, pos):
        radius = self.lidar_reach

        fig, ax = plt.subplots()

        for key in self.grid.keys():
            x, y = key 
            # plt.plot(x, y, '-o', color='red') # plot real cell corner
            ax.plot(x + self.cell_dim/2, y + self.cell_dim/2, '-o', color='black') # plot on the middle because it looks better
        
        plt.setp(ax, xticks=np.arange(int(pos[0] - radius - 2*self.cell_dim), int(pos[0] + 2*radius + 2*self.cell_dim), self.cell_dim),
        yticks=np.arange(int(pos[1] - radius - 2*self.cell_dim), int(pos[1] + radius + 2*self.cell_dim), self.cell_dim))

        ax.add_patch(plt.Circle(xy=(pos[0], pos[1]), radius=10, color='b', fill=False))
        ax.grid()

        return fig
    
    def update_grid(self, pos, measure):
        for angle, data in measure.items():
            self.mark_cells_in_ray(pos=pos, angle=angle, measurement=data)
    
    def create_eight_conn_edges(self, node):
        directions = [(-self.cell_dim, -self.cell_dim), (-self.cell_dim, 0), (-self.cell_dim, self.cell_dim), 
                  ( 0, -self.cell_dim),         ( 0, self.cell_dim), 
                  ( self.cell_dim, -self.cell_dim), ( self.cell_dim, 0), ( self.cell_dim, self.cell_dim)]
        
        node = np.array(node)
        
        for direction in directions:
            direction = np.array(direction)
            neighbour = node + direction
            if not self.graph.has_edge(tuple(node), tuple(neighbour)) and self.graph.has_node(tuple(neighbour)):
                self.graph.add_edge(tuple(node), tuple(neighbour), weight=np.linalg.norm(neighbour - node))

    
    def update_graph(self):
        for cell, _ in self.grid.items():
            if (self.grid[cell] == CellInfo.FREE) and (not self.graph.has_node(cell)):
                self.graph.add_node(cell)
        
        for node in self.graph.nodes():
            self.create_eight_conn_edges(node)