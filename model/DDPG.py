import random
import numpy as np

class DDPGNetwork():

    def __init__(self, gamma, actor_model, actor_target_model, critic_model, critic_target_model, batch_sz, episode_state_history_max=10000,
                 episode_state_history_min=100, update_period=1):
        self.episode_state_history_max = episode_state_history_max
        self.episode_state_history_min = episode_state_history_min
        self.episode_state_history = list()
        self.batch_sz = batch_sz

        self.actor_model = actor_model
        self.actor_target_model = actor_target_model

        self.critic_model = critic_model
        self.critic_target_model = critic_target_model

        self.gamma = gamma

        self.train_counter = 0
        self.update_period = update_period

    def train(self):
        """
        Will train if there is enough episode history.
        Will update the target networks at the interval given by self.update_period
        :return:
        """

        episode_state_history_size = len(self.episode_state_history)
        if episode_state_history_size > self.episode_state_history_min:

            training_sample = random.sample(self.episode_state_history, self.batch_sz)

            states, actions, rewards, next_states = map(np.array, zip(*training_sample))

            target_next_actions = self.actor_target_model.predict(next_states)
            target_q_values = self.critic_target_model.predict(next_states, target_next_actions)

            predicted_q  = np.reshape(rewards, (-1,1)) + self.gamma * target_q_values

            cost, _ = self.critic_model.partial_fit(predicted_q, states, actions)

            next_action_for_gradient = self.actor_model.predict(states)
            q_gradients = self.critic_model.gradients(states, next_action_for_gradient)

            self.actor_model.partial_fit(q_gradients, states)

            if self.train_counter % self.update_period == 0:
                #print("Updating target network at %d steps" % self.train_counter)
                self.update_target_network(self.actor_model, self.actor_target_model)
                self.update_target_network(self.critic_model, self.critic_target_model)

            self.train_counter += 1
            return cost

        return 0.0


    def update_target_network(self, model, target_model):
        params = model.get_params()
        target_model.update_target_params(params)

    def reset_experience(self):
        self.episode_state_history = list()

    def add_experience(self, experience):
        if len(self.episode_state_history) > self.episode_state_history_max:
            self.episode_state_history.pop(0)
        self.episode_state_history.append(experience)

    def sample_action(self, state, epsilon):
        return self.actor_model.sample_action(state, epsilon)

    def set_session(self,session):
        self.actor_model.set_session(session)
        self.actor_target_model.set_session(session)

        self.critic_model.set_session(session)
        self.critic_target_model.set_session(session)