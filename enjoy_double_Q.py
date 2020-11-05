import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import random
from matplotlib import pyplot as plt
from tensorflow import convert_to_tensor
import tensorflow as tf

# CARTPOLE GAME SETTINGS
OBSERVATION_SPACE_DIMS = 4
ACTION_SPACE = [0, 1]

# AGENT/NETWORK HYPERPARAMETERS
EPSILON_INITIAL = 0.5  # exploration rate
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
ALPHA = 0.001  # learning rate
GAMMA = 0.99  # discount factor
TAU = 0.1  # target network soft update hyperparameter
EXPERIENCE_REPLAY_BATCH_SIZE = 32
AGENT_MEMORY_LIMIT = 2000
MIN_MEMORY_FOR_EXPERIENCE_REPLAY = 500


def create_dqn():
    # not actually that deep
    nn = Sequential()
    nn.add(Dense(64, input_dim=OBSERVATION_SPACE_DIMS, activation='relu'))
    nn.add(Dense(64, activation='relu'))
    nn.add(Dense(len(ACTION_SPACE), activation='linear'))
    nn.compile(loss='mse', optimizer=Adam(lr=ALPHA))
    return nn


def create_bc():
    # not actually that deep
    nn = Sequential()
    nn.add(Dense(64, input_dim=OBSERVATION_SPACE_DIMS, activation='relu'))
    nn.add(Dense(64, activation='relu'))
    nn.add(Dense(len(ACTION_SPACE), activation='softmax'))
    nn.compile(loss='categorical_crossentropy', optimizer=Adam(lr=ALPHA))
    return nn


class DoubleDQNAgent(object):

    def __init__(self):
        self.memory = []
        self.online_network = create_dqn()
        self.target_network = create_dqn()
        # Create clone agent
        self.bc_network = create_dqn()
        self.threshold = 0
        self.epsilon = EPSILON_INITIAL
        self.has_talked = False

    def act(self, state):
        if self.epsilon > np.random.rand():
            # explore
            return np.random.choice(ACTION_SPACE)
        else:
            # exploit
            state = self._reshape_state_for_net(state)
            q_values = self.online_network.predict(state)[0]
            return np.argmax(q_values)

    def select_action(self, next_state):

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0, 1) > 0.001:
            q_val = self.online_network.predict(next_state)[0]
            #ratio = q_val / q_val[np.argmax(q_val)]
            ratio = self.bc_network.predict(next_state)[0] / np.argmax(self.bc_network.predict(next_state)[0])
            # Use large negative number to mask actions from argmax
            return np.argmax(ratio)
        else:
            return np.random.randint(len(ACTION_SPACE))

    def experience_replay(self):

        minibatch = random.sample(self.memory, EXPERIENCE_REPLAY_BATCH_SIZE)
        minibatch_new_q_values = []

        # 4. Sample mini-batch M of N transitions (s,a,r,s') from B
        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = self._reshape_state_for_net(state)
            experience_new_q_values = self.online_network.predict(state)[0]
            if done:
                q_update = reward
            else:
                next_state = self._reshape_state_for_net(next_state)
                # using online network to SELECT action
                # 5. a' = argmax_{a' | pi(a'|s') / max_{a} pi(a | s') > tau} Q(s',a') <-- Implement this
                #online_net_selected_action = np.argmax(self.online_network.predict(next_state))
                online_net_selected_action = self.select_action(next_state)
                # using target network to EVALUATE action
                target_net_evaluated_q_value = self.target_network.predict(next_state)[0][online_net_selected_action]
                # Build the target prediction for 6.
                q_update = reward + GAMMA * target_net_evaluated_q_value
            experience_new_q_values[action] = q_update
            # Collect into mini-batch for 6.
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.array([e[0] for e in minibatch])
        minibatch_new_q_values = np.array(minibatch_new_q_values)
        # 6. Update parameters for batch (r + gamma * Q_target - Q_online)
        self.online_network.fit(minibatch_states, minibatch_new_q_values, verbose=False, epochs=1)
        # 7. Update parameters for (s,a,_,_) in batch for log(pi(a | s)) <-- Implement this
        self.bc_update()

    def bc_update(self):

        minibatch = random.sample(self.memory, EXPERIENCE_REPLAY_BATCH_SIZE)
        #minibatch_new_q_values = []
        minibatch_actions = []

        # 4. Sample mini-batch M of N transitions (s,a,r,s') from B
        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = self._reshape_state_for_net(state)
            #experience_new_q_values[action] = q_update
            #experience_actions = action
            # Collect into mini-batch for 6.
            #minibatch_new_q_values.append(experience_new_q_values)
            minibatch_actions.append(action)
        minibatch_states = np.array([e[0] for e in minibatch])
        #minibatch_new_q_values = np.array(minibatch_new_q_values)
        minibatch_actions = np.array(minibatch_actions)
        # 6. Update parameters for batch (r + gamma * Q_target - Q_online)
        # self.online_network.fit(minibatch_states, minibatch_new_q_values, verbose=False, epochs=1)
        # 7. Update parameters for (s,a,_,_) in batch for log(pi(a | s)) <-- Implement this

        self.bc_network.fit(minibatch_states,to_categorical(minibatch_actions, len(ACTION_SPACE)), verbose = False, epochs = 1)

    def update_target_network(self):
        q_network_theta = self.online_network.get_weights()
        target_network_theta = self.target_network.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta, target_network_theta):
            target_weight = target_weight * (1 - TAU) + q_weight * TAU
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_network.set_weights(target_network_theta)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) <= AGENT_MEMORY_LIMIT:
            experience = (state, action, reward, next_state, done)
            self.memory.append(experience)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)

    def _reshape_state_for_net(self, state):
        return np.reshape(state, (1, OBSERVATION_SPACE_DIMS))


def test_agent(flag = False):
    env = gym.make('CartPole-v0')
    env.seed(1)
    trials = []
    NUMBER_OF_TRIALS = 1
    MAX_TRAINING_EPISODES = 2000
    MAX_STEPS_PER_EPISODE = 200

    for trial_index in range(NUMBER_OF_TRIALS):
        agent = DoubleDQNAgent()
        trial_episode_scores = []

        for episode_index in range(1, MAX_TRAINING_EPISODES + 1):
            state = env.reset()
            episode_score = 0

            for _ in range(MAX_STEPS_PER_EPISODE):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                episode_score += reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if len(agent.memory) > MIN_MEMORY_FOR_EXPERIENCE_REPLAY:
                    if not flag:
                        agent.experience_replay()
                        agent.update_target_network()
                        agent.bc_update()
                    else:
                        agent.bc_update()
                if done:
                    break

            trial_episode_scores.append(episode_score)
            agent.update_epsilon()
            last_100_avg = np.mean(trial_episode_scores[-100:])
            print('E %d scored %d, avg %.2f' % (episode_index, episode_score, last_100_avg))
            if len(trial_episode_scores) >= 100 and last_100_avg >= 195.0:
                print('Trial %d solved in %d episodes!' % (trial_index, (episode_index - 100)))
                break
        trials.append(np.array(trial_episode_scores))
    return np.array(trials)


def plot_trials(trials):
    _, axis = plt.subplots()

    for i, trial in enumerate(trials):
        steps_till_solve = trial.shape[0] - 100
        # stop trials at 2000 steps
        if steps_till_solve < 1900:
            bar_color = 'b'
            bar_label = steps_till_solve
        else:
            bar_color = 'r'
            bar_label = 'Stopped at 2000'
        plt.bar(np.arange(i, i + 1), steps_till_solve, 0.5, color=bar_color, align='center', alpha=0.5)
        axis.text(i - .25, steps_till_solve + 20, bar_label, color=bar_color)

    plt.ylabel('Episodes Till Solve')
    plt.xlabel('Trial')
    trial_labels = [str(i + 1) for i in range(len(trials))]
    plt.xticks(np.arange(len(trials)), trial_labels)
    # remove y axis labels and ticks
    axis.yaxis.set_major_formatter(plt.NullFormatter())
    plt.tick_params(axis='both', left='off')

    plt.title('Double DQN CartPole v-0 Trials')
    plt.show()


def plot_individual_trial(trial):
    plt.plot(trial)
    plt.ylabel('Steps in Episode')
    plt.xlabel('Episode')
    plt.title('Double DQN CartPole v-0 Steps in Select Trial')
    plt.show()


if __name__ == '__main__':
    trials = test_agent()
    # print 'Saving', file_name
    # np.save('double_dqn_cartpole_trials.npy', trials)
    # trials = np.load('double_dqn_cartpole_trials.npy')
    plot_trials(trials)
    plot_individual_trial(trials[1])