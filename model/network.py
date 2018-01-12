import tensorflow as tf
import uuid

class FeedFoward(object):

    def __init__(self, n_inputs, n_outputs, layers, name=None):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.param_copy_ops = []

        if name:
            self.scope = name
        else:
            self.scope = str(uuid.uuid4())

        self.create_network(layers)

    def create_network(self, layers):
        with tf.variable_scope(self.scope):
            self.network_input = tf.placeholder("float", [None, self.n_inputs])
            self._create_layers(layers, self.network_input)

    def _create_layers(self, layers, input):

        prev_n_inputs = self.n_inputs

        prev_layer = input

        for i, (n, activation, has_bias) in enumerate(layers):

            # Create variables for the weights and the bias
            W = tf.get_variable("W_%d" % i, shape=(prev_n_inputs, n), initializer=tf.contrib.layers.xavier_initializer())
            layer_output = tf.matmul(prev_layer, W)
            if has_bias:
                b = tf.get_variable("b_%d" % i, shape=(n), initializer=tf.contrib.layers.xavier_initializer())
                prev_layer = tf.add(layer_output, b)
            else:
                prev_layer = layer_output

            prev_layer = activation(prev_layer)

            prev_n_inputs = n


        self.network_output = prev_layer

    def set_target_param_op(self, current_param, new_param_placeholder):
        """
        Default set_param operation just sets the values that are in the new param placeholder,

        This can be replaced with other operations such as exp moving average etc...
        :param current_param:
        :param new_param_placeholder:
        :return:
        """
        return current_param.assign(new_param_placeholder)

    def create_target_params_update_operation(self):
        """
        create a graph here that allows the parameters of the entire network to be set
        :return:
        """
        param_copy_ops = []
        for current_param in self.get_params():
            new_param_placeholder = tf.placeholder(tf.float32, shape=current_param.shape)
            param_copy_ops.append((self.set_target_param_op(current_param, new_param_placeholder), new_param_placeholder))

        return param_copy_ops

    def update_target_params(self, new_params):
        for copy_op, input in zip(self.param_copy_ops, new_params):
            self.session.run(copy_op[0], feed_dict={ copy_op[1]: self.session.run(input) })

    def get_params(self):
        return sorted([t for t in tf.trainable_variables() if t.name.startswith(self.scope)], key=lambda v: v.name)

    def set_session(self, session):
        self.session = session


