import numpy as np

from logger import setup_custom_logger
from embod_client import AsyncClient
from tensorforce.agents import PPOAgent

class Controller:

    def __init__(self, apikey, agent_id, host=None):

        self._agent = PPOAgent(
            states_spec=dict(type='float', shape=(14,)),
            actions_spec=dict(type='float',
                              shape=(3,),
                              min_value=np.float32(-1.),
                              max_value=np.float32(1.)),
            network_spec=[
                dict(type='dense', activation='relu', size=64),
                dict(type='dense', activation='relu', size=64),
            ],
            optimization_steps=5,
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
                learning_rate=5e-4
            )
        )

        self._logger = setup_custom_logger("Controller")

        self._frame_count_per_episode = 0
        self._total_frames = 0

        self._client = AsyncClient(apikey, agent_id, self._train_state_callback, host)

    async def _train_state_callback(self, state, reward, error):

        terminal = False

        if reward is not 0.0:
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

        action = self._agent.act(state[11:])

        await self._client.send_agent_action(action)

        self._total_frames += 1
        self._frame_count_per_episode += 1

        if self._total_frames % 100 == 0:

            self._logger.info("%d iterations: AVG reward: %.2f" %
                             (
                                 self._total_frames,
                                 self.total_rewards[max(0, self._total_frames - 100):self._total_frames].mean())
                             )

        if self._total_frames >= self.max_iterations:
            self._client.stop()

    def _run_state_callback(self, message_type, resource_id, state, reward, error):
        pass


    def train(self, max_iterations, max_frame_count_per_episode=1000):


        self._max_frame_count_per_episode = max_frame_count_per_episode

        self.max_iterations = max_iterations
        self.total_rewards = np.zeros(max_iterations)
        self.total_costs = np.zeros(max_iterations)


        self._client.start()


    def run(self):


        self._client.start()