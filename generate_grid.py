import numpy as np
import matplotlib.pyplot as plt

# make a simple grid data
class GridData:
    def __init__(self, x_max, y_max, num_shapes=5, shape_max_size=10):
        self.x_max = x_max
        self.y_max = y_max
        self.num_shapes = num_shapes
        self.shape_max_size = shape_max_size
    
    def generate_grid(self):
        # randomly generate shapes and put into the grid
        # shape can be a circle, a square, a triangle
        self.grid = np.zeros((self.x_max, self.y_max))
        for i in range(self.num_shapes):
            shape = np.random.randint(3)
            if shape == 0:
                self.grid = self.draw_circle(self.grid)
            elif shape == 1:
                self.grid = self.draw_square(self.grid)
            elif shape == 2:
                self.grid = self.draw_triangle(self.grid)
        
        return self.grid
    
    def draw_circle(self, grid):
        # randomly generate a circle
        x = np.random.randint(self.x_max-1)
        y = np.random.randint(self.y_max-1)
        r = np.random.randint(1, min(self.x_max-x, self.y_max-y))
        r = min(r, self.shape_max_size)
        for i in range(self.x_max):
            for j in range(self.y_max):
                if (i-x)**2 + (j-y)**2 <= r**2:
                    grid[i][j] = 1
        return grid
    
    def draw_square(self, grid):
        # randomly generate a square
        x = np.random.randint(self.x_max)
        y = np.random.randint(self.y_max)
        r = np.random.randint(1, min(self.x_max-x, self.y_max-y))
        r = min(r, self.shape_max_size)
        for i in range(self.x_max):
            for j in range(self.y_max):
                if abs(i-x) <= r and abs(j-y) <= r:
                    grid[i][j] = 1
        return grid
    
    def draw_triangle(self, grid):
        # randomly generate a triangle
        x = np.random.randint(self.x_max)
        y = np.random.randint(self.y_max)
        r = np.random.randint(1, min(self.x_max-x, self.y_max-y))
        r = min(r, self.shape_max_size)
        for i in range(self.x_max):
            for j in range(self.y_max):
                if abs(i-x) + abs(j-y) <= r:
                    grid[i][j] = 1
        return grid
    
if __name__ == '__main__':
    Grid = GridData(128, 128, num_shapes=15)
    grid = Grid.generate_grid()
    plt.imshow(grid)
    plt.show()


