import tensorflow as tf
import numpy as np

from model.network import Layers

class ActorNetwork(Layers):

    def __init__(self, session, n_inputs, n_outputs, layers):
        Layers.__init__(self, session, n_inputs, n_outputs, layers)


    def train(self):