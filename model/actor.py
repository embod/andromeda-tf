import tensorflow as tf

from model.network import FeedFoward
from model.ou_noise import OUNoise

class ActorNetwork(FeedFoward):


    def __init__(self, n_inputs, n_outputs, layers, learning_rate=None, target_ema_decay=None, name=None):
        FeedFoward.__init__(self, n_inputs, n_outputs, layers, name)

        self.learning_rate = learning_rate
        self.target_ema_decay = target_ema_decay

        # If this is a target network we dont need to create these operations
        if learning_rate is not None:
            self.create_train_op()
            self.exploration_noise = OUNoise(self.n_outputs)

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