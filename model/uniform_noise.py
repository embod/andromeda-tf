import numpy as np
import numpy.random as nr

class AverageUniformNoise:
    """
    Create Random uniform noise
    """
    def __init__(self, action_dimension, decay=0.9, alpha=0.1):
        self._action_dimension = action_dimension

        # Initialize the state
        self._state = nr.uniform(-0.4, 0.4, self._action_dimension)

        self._decay = decay
        self._alpha = alpha
        self.reset()

    def reset(self):
        pass

    def noise(self, epsilon):
        self._state = nr.uniform(-self._alpha, self._alpha, self._action_dimension) + self._decay*self._state

        return epsilon*self._state
