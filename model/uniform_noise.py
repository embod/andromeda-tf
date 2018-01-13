import numpy as np
import numpy.random as nr

class AverageUniformNoise:
    """
    Create Random uniform noise
    """
    def __init__(self, action_dimension):
        self.action_dimension = action_dimension
        self.state = nr.uniform(-0.4, 0.4, self.action_dimension)
        self.reset()

    def reset(self):
        pass

    def noise(self):
        self.state = nr.uniform(-0.1, 0.1, self.action_dimension) + 0.9*self.state
        return self.state
