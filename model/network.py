import tensorflow as tf
import numpy as np

class Layers(object):
    """

    """

    def __init__(self, session, n_inputs, n_outputs, layers):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.session = session

    def create_network(self, layers):
        self.create_layers(layers)

    def create_layers(self, layers):

        prev_n_inputs = self.n_inputs
        self.network_input = tf.placeholder("float", [None,self.n_inputs])

        prev_layer = input

        for i, (n, activation, has_bias) in enumerate(layers):

            # Create variables for the weights and the bias
            W = tf.Variable("W_%d" % i, (prev_n_inputs, n))
            layer_output = tf.matmul(prev_layer, W)
            if has_bias:
                b = tf.Variable("b_%d" % i, n)
                prev_layer = tf.add(layer_output, b)
            else:
                prev_layer = layer_output

            prev_n_inputs = n

        self.network = prev_layer


    def predict(self, input):

        self.session.run(self.network, feed_dict={
			self.network_input:input
			})

    def set_parameters(self):


