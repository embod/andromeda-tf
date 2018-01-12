import tensorflow as tf

from model.network import FeedFoward

class CriticNetwork(FeedFoward):
    LEARNING_RATE = 0.001
    L2 = 0.01
    DECAY = 0.999

    def __init__(self, n_action_inputs, n_state_inputs, layers, learning_rate=None, target_ema_decay=None, name=None):

        self.n_action_inputs = n_action_inputs
        self.n_state_inputs = n_state_inputs

        FeedFoward.__init__(self, layers[1][0], 1, layers, name)

        # If this is a target network we dont need to create these operations
        if learning_rate is not None:
            self.create_train_op()

        if target_ema_decay is not None:
            self.create_target_params_update_operation()

    def create_network(self, layers):
        """
        Create the critic network
        :param layers:
        :return:
        """

        self.network_state_input = tf.placeholder("float", [None, self.n_state_inputs])
        self.network_action_input = tf.placeholder("float", [None, self.n_action_inputs])

        state_layer_n_outputs, state_layer_activation, state_layer_has_bias = layers[0]
        action_layer_n_outputs, action_layer_activation, action_layer_has_bias = layers[1]

        with tf.variable_scope(self.scope):

            W1 = tf.get_variable("W1", shape=(self.n_state_inputs, state_layer_n_outputs), initializer=tf.contrib.layers.xavier_initializer())
            state_layer_output = tf.matmul(self.network_state_input, W1)
            if state_layer_has_bias:
                b1 = tf.get_variable("b1", shape=(state_layer_n_outputs), initializer=tf.contrib.layers.xavier_initializer())
                state_layer_output = tf.add(state_layer_output, b1)

            W2 = tf.get_variable("W2", shape=(state_layer_n_outputs, action_layer_n_outputs), initializer=tf.contrib.layers.xavier_initializer())

            W_action = tf.get_variable("W_action", shape=(self.n_action_inputs, action_layer_n_outputs), initializer=tf.contrib.layers.xavier_initializer())
            action_layer_output = tf.matmul(self.network_action_input, W_action) +tf.matmul(state_layer_output, W2)
            if action_layer_has_bias:
                b_action = tf.get_variable("b_action", action_layer_n_outputs, initializer=tf.contrib.layers.xavier_initializer())
                action_layer_output = tf.add(action_layer_output, b_action)

            self._create_layers(layers[2:], action_layer_output)

    def create_train_op(self):
        """
        Creates the operations for training
        :return:
        """

        self.predicted_q_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([CriticNetwork.L2 * tf.nn.l2_loss(var) for var in self.get_params()])
        self.cost = tf.reduce_mean(tf.square(self.predicted_q_input - self.network_output)) + weight_decay
        self.train_op = tf.train.AdamOptimizer(CriticNetwork.LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.network_output, self.network_action_input)

    def set_target_param_op(self, current_param, new_param_placeholder):
        """
        Exponential moving average update
        :param current_param:
        :param new_param_placeholder
        :return:
        """
        ema_update_value = current_param - (1 - CriticNetwork.DECAY) * (current_param - new_param_placeholder)
        return tf.assign_sub(current_param, ema_update_value)

    def predict(self, state_input, action_input):
        return self.session.run(self.network_output,
            feed_dict = {
                self.network_state_input: state_input,
                self.network_action_input: action_input
            }
        )

    def gradients(self, states, actions):
        """
        returns the gradients of the action input with respect to the critic output network
        :param states:
        :param actions:
        :return:
        """
        return self.session.run((
                self.action_gradients,
            ),
            feed_dict={
                self.network_state_input: states,
                self.network_action_input: actions
            }
        )

    def partial_fit(self, predicted_qs, states, actions):
        return self.session.run(
            (
                self.cost,
                self.train_op
            ),
            feed_dict={
                self.predicted_q_input: predicted_qs,
                self.network_state_input: states,
                self.network_action_input: actions
            }
        )