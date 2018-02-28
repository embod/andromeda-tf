import numpy as np

class StateStack:

    def __init__(self, max_states=5):
        self._state_stack = []
        self._max_states = max_states

    def add_state(self, state):
        if len(self._state_stack) == self._max_states:
            del self._state_stack[0]
        self._state_stack.append(state)

    def split_means(self):
        means = np.mean(np.array(self._state_stack), axis=0)

        mbot_sensor = means[0:11]
        food_sensor = means[11:22]
        poison_sensor = means[22:33]

        xvelocity = means[33]
        yvelocity = means[34]
        zvelocity = means[35]

        return mbot_sensor, food_sensor, poison_sensor, xvelocity, yvelocity, zvelocity

    def get_combined_state(self):
        return np.hstack(self._state_stack)