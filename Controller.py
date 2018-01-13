import tensorflow as tf
import numpy as np
from uuid import UUID
import logging

from model import DDPGNetwork
from model import CriticNetwork
from model import ActorNetwork
from embodsdk import Client

class Controller:

    def __init__(self, apikey, agent_id):

        self.logger = logging.getLogger(__name__)

        self.agent_id = UUID(agent_id)
        self.apikey = apikey

        self.actor_layers = [
            #(256, tf.nn.relu, True),
            (128, tf.nn.sigmoid, True),
            (64, tf.nn.sigmoid, True),
            (32, tf.nn.sigmoid, True)
        ]

        self.critic_layers = [
            #(256, tf.nn.relu, True),
            (128, tf.nn.sigmoid, True),
            (64, tf.nn.sigmoid, True),
            (32, tf.nn.sigmoid, True),
            (1, tf.identity, True)
        ]

        self.num_actions = 3
        self.num_states = 22

        self.gamma = 0.99

        self.actor_model = ActorNetwork(
            self.num_states, self.num_actions, self.actor_layers, learning_rate=0.0001, name="actor_model")

        self.actor_target_model = ActorNetwork(
            self.num_states, self.num_actions, self.actor_layers, target_ema_decay=0.999, name="actor_target_model")

        self.critic_model = CriticNetwork(
            self.num_actions, self.num_states, self.critic_layers, learning_rate=0.0001, name="critic_model")

        self.critic_target_model = CriticNetwork(
            self.num_actions, self.num_states, self.critic_layers, target_ema_decay=0.999, name="critic_target_model")

        self.ddpg = DDPGNetwork(
            self.gamma,
            self.actor_model,
            self.actor_target_model,
            self.critic_model,
            self.critic_target_model,
            64,
            episode_state_history_max=10000,
            episode_state_history_min=1000

        )

        self.prev_state = None

        init = tf.global_variables_initializer()
        session = tf.Session()

        self.ddpg.set_session(session)

        session.run(init)
        session.graph.finalize()

    def _train_state_callback(self, resource_id, state, reward, error):

        if error:
            self.logger.error(error)
            return

        if self.prev_state is None:
            self.prev_state = state
            return

        next_action = np.reshape(self.ddpg.sample_action(np.atleast_2d(state)), self.num_actions)

        self.ddpg.train()

        self.ddpg.add_experience([self.prev_state, next_action, reward, state])
        cost = self.ddpg.train()

        self.total_rewards[self.iterations] = reward
        self.total_costs[self.iterations] = cost

        self.client.send_agent_action(resource_id, next_action)

        self.prev_state = state

        self.iterations += 1


        if self.iterations >= self.max_iterations:
            self.client.stop()

    def _run_state_callback(self, message_type, resource_id, state, reward, error):
        pass


    def train(self, max_iterations):
        self.max_iterations = max_iterations
        self.iterations = 0
        self.total_rewards = np.zeros(max_iterations)
        self.total_costs = np.zeros(max_iterations)

        self.client = Client(self.apikey, self._train_state_callback)
        self.client.add_agent(self.agent_id)

        self.client.run_loop()


    def run(self):

        self.client = Client(self.apikey, self._run_state_callback)
        self.client.add_agent(self.agent_id)

        self.client.run_loop()