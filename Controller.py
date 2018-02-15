import numpy as np

from logger import setup_custom_logger
from embod_client import AsyncClient
from tensorforce.agents import PPOAgent

class Controller:

    def __init__(self, apikey, agent_id, host=None):

        # PPO agent seems to learn that it needs to speed around the environment to collect rewards
        self._agent = PPOAgent(
            states_spec=dict(type='float', shape=(25,)),
            actions_spec=dict(type='float',
                              shape=(3,),
                              min_value=np.float32(-1.0),
                              max_value=np.float32(1.0)),
            network_spec=[
                dict(type='dense', activation='relu', size=500),
                dict(type='dense', activation='relu', size=500),
            ],
            optimization_steps=10,
            # Model
            scope='ppo',
            discount=0.99,
            # DistributionModel
            distributions_spec=None,
            entropy_regularization=0.01,
            # PGModel
            baseline_mode=None,
            baseline=None,
            baseline_optimizer=None,
            gae_lambda=None,
            # PGLRModel
            likelihood_ratio_clipping=0.2,
            summary_spec=None,
            distributed_spec=None,
            batch_size=2048,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-4
            )
        )

        self._logger = setup_custom_logger("Controller")

        self._frame_count_per_episode = 0
        self._total_frames = 0

        self._client = AsyncClient(apikey, agent_id, self._train_state_callback, host)

    async def _train_state_callback(self, state, reward, error):

        terminal = False

        # We are controlling the the episode to be terminal if either
        # 1. the agent gets a reward in the environment
        # 2. the agent has not had a reward for _frame_count_per_episode states from the environment
        if reward != 0.0:
            reward = reward * 20.0
            terminal = True
            self._frame_count_per_episode = 0
            print("terminal, got reward - %.2f" % reward)
        elif self._frame_count_per_episode == self._max_frame_count_per_episode:
            reward = -100.0
            terminal = True
            self._frame_count_per_episode = 0
            print("terminal, killing")

        if self._total_frames > 0:
            self._agent.observe(reward=reward, terminal=terminal)

        # Currently ignoring the first 11 states as they are sensor for other agents in the environment
        action = self._agent.act(state[11:])

        # Only let the mbot travel forwards
        action[0] = (action[0]+1.0)/2.0

        self.total_rewards[self._total_frames] = reward

        await self._client.send_agent_action(action)

        self._total_frames += 1
        self._frame_count_per_episode += 1

        if self._total_frames % 100 == 0:

            self._logger.info("%d iterations: Running AVG reward per last %d states: %.2f" %
                             (
                                 self._total_frames,
                                 self._max_frame_count_per_episode,
                                 self.total_rewards[max(0, self._total_frames - 10000):self._total_frames].mean())
                             )

        if self._total_frames >= self.max_iterations:
            self._client.stop()

    def train(self, max_iterations, max_frame_count_per_episode=1000):
        """
        :param max_iterations: the maximum iterations across all episodes
        :param max_frame_count_per_episode: we control how the episodes are handled
        :return:
        """
        self._max_frame_count_per_episode = max_frame_count_per_episode

        self.max_iterations = max_iterations
        self.total_rewards = np.zeros(max_iterations)
        self.total_costs = np.zeros(max_iterations)

        self._client.start()
