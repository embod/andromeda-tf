import tensorflow as tf

from model.network import FeedFoward
from model.ou_noise import OUNoise
from model.uniform_noise import AverageUniformNoise

class ActorNetwork(FeedFoward):


    def __init__(self, n_inputs, n_outputs, layers, learning_rate=None, target_ema_decay=None, name=None):
        FeedFoward.__init__(self, n_inputs, n_outputs, layers, name)

        last_layer_size = layers[-1][0]

        W_final = tf.Variable(tf.random_uniform([last_layer_size, n_outputs], -3e-3, 3e-3))
        b_final = tf.Variable(tf.random_uniform([n_outputs], -3e-3, 3e-3))

        self.network_output = tf.tanh(tf.matmul(self.network_output, W_final) + b_final)

        self.learning_rate = learning_rate
        self.target_ema_decay = target_ema_decay

        # If this is a target network we dont need to create these operations
        if learning_rate is not None:
            self.create_train_op()
            #self.exploration_noise = OUNoise(self.n_outputs, mu=0, sigma=0.01, theta=1)
            self.exploration_noise = AverageUniformNoise(self.n_outputs)

        if target_ema_decay is not None:
            self.create_target_params_update_operation()

    def create_train_op(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.n_outputs])
        self.parameters_gradients = tf.gradients(self.network_output, self.get_params(), -self.q_gradient_input)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.parameters_gradients, self.get_params()))

    def partial_fit(self, q_gradient_inputs, network_inputs):
        return self.session.run(self.train_op,
            feed_dict={
                self.q_gradient_input: q_gradient_inputs,
                self.network_input: network_inputs
            }
        )

    def set_target_param_op(self, current_param, new_param_placeholder):
        """
        Exponential moving average update
        :param current_param:
        :param new_param_placeholder:
        :return:
        """
        ema_update_value = current_param - (1 - self.target_ema_decay) * (current_param - new_param_placeholder)
        return tf.assign_sub(current_param, ema_update_value)


    def sample_action(self, state_input):
        return self.predict(state_input) + self.exploration_noise.noise()

    def predict(self, state_input):
        return self.session.run(self.network_output,
            feed_dict = {
                self.network_input: state_input
            }
        )

    def reset_exploration_noise(self):
        self.exploration_noise.reset()