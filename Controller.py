import tensorflow as tf
import numpy as np
from uuid import UUID
import logging

from model import DDPGNetwork
from model import CriticNetwork
from model import ActorNetwork
from embodsdk import AsyncClient

class Controller:

    def __init__(self, apikey, agent_id):

        self.logger = logging.getLogger("root")

        self.agent_id = UUID(agent_id)
        self.apikey = apikey

        self.actor_layers = [
            #(256, tf.nn.relu, True),
            #(128, tf.nn.relu, True),
            (64, tf.nn.sigmoid, True),
            (32, tf.nn.sigmoid, True)
        ]

        self.critic_layers = [
            #(256, tf.nn.relu, True),
            #(128, tf.nn.relu, True),
            (64, tf.nn.sigmoid, True),
            (32, tf.nn.sigmoid, True),
            (1, tf.identity, True)
        ]

        self.num_actions = 3
        self.num_states = 22

        self.gamma = 0.99
        self.epsilon = 1.0
        self.i = 0

        self.actor_model = ActorNetwork(
            self.num_states, self.num_actions, self.actor_layers, learning_rate=0.00001, name="actor_model")

        self.actor_target_model = ActorNetwork(
            self.num_states, self.num_actions, self.actor_layers, target_ema_decay=0.999, name="actor_target_model")

        self.critic_model = CriticNetwork(
            self.num_actions, self.num_states, self.critic_layers, learning_rate=0.005, name="critic_model")

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

    async def _train_state_callback(self, state, reward, error):

        if error:
            self.logger.error(error)
            return

        if self.prev_state is None:
            self.prev_state = state
            return

        if reward == 0.0:
            reward = -0.1
        if reward == 1.0:
            reward = 100.0

        next_action = np.reshape(self.ddpg.sample_action(np.atleast_2d(state), epsilon=self.epsilon), self.num_actions)

        self.ddpg.train()

        self.ddpg.add_experience([self.prev_state, next_action, reward, state])
        cost = self.ddpg.train()

        self.total_rewards[self.i] = reward
        self.total_costs[self.i] = cost

        await self.client.send_agent_action(next_action)

        self.prev_state = state

        self.i += 1

        if self.i % 100 == 0:
            if self.epsilon > 0.0:
                self.epsilon -= 0.001
            else:
                self.epsilon = 0.0
            self.logger.info("%d iterations: AVG reward: %.2f, AVG cost: %.2f, epsilon: %.3f" %
                             (self.i,
                              self.total_rewards[max(0, self.i - 100):self.i].mean(),
                              self.total_costs[max(0, self.i - 100):self.i].mean(),
                              self.epsilon))

        if self.i >= self.max_iterations:
            self.client.stop()

    def _run_state_callback(self, message_type, resource_id, state, reward, error):
        pass


    def train(self, max_iterations):
        self.max_iterations = max_iterations
        self.total_rewards = np.zeros(max_iterations)
        self.total_costs = np.zeros(max_iterations)

        self.client = AsyncClient(self.apikey, self.agent_id, self._train_state_callback)

        self.client.start()


    def run(self):

        self.client = AsyncClient(self.apikey, self._run_state_callback)

        self.client.start()