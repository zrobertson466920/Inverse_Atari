# Environment
import gym
#from gym.spaces import prng

# Utility
import cv2
import numpy as np
from collections import deque
import argparse
from functools import partial
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import tensorflow as tf

from models import clipped_mse


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Rescales the Image width by k and the height by l.
def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


# Makes an environment for simulation
# Should be exact same every time now
def make_environment(game):
    env = gym.make(game)
    env = MaxAndSkipEnv(env, 2)
    env.seed(0)
    env.action_space.np_random.seed(0)
    #prng.seed(0)
    return env


# Makes a simple catalog for an episode ~ [observation,action,reward*,done*,info*]
# *Unless initial state
# Action Effects: 0 ~ None, 1 ~ Fire, 2 ~ Right, 3 ~ Left
def record_episode(env, num=1):
    episodes = []
    for i in range(num):
        done = False
        lives = 5
        t_lives = 5
        episode = []
        episode.append([env.reset(), 0, done, None])
        while not done:
            if lives - t_lives is True:
                action = 1
                episode[-1].insert(1, action)
                episode.append(list(env.step(action)))
                done = episode[-1][2]
                lives = t_lives
                t_lives = episode[-1][-1]['ale.lives']
            else:
                action = env.action_space.sample()
                episode[-1].insert(1, action)
                episode.append(list(env.step(action)))
                done = episode[-1][2]
                t_lives = episode[-1][-1]['ale.lives']
        episodes.append(episode)
    env.close()
    return episodes, env.action_space.n


# Record a human playing the game Breakout
# Same format as record_episode
def record_human(ep, eps, obs_before, obs_after, action, rew, done, info):
    if done is False:
        ep.append((obs_before, action, rew, done))
    else:
        eps.append(ep)
        ep = []
    return


# Save episodes into directory sorted by episode number
# Caution: Will overwrite without hesitation
def save_episodes(dir, eps, num=0):
    for i in range(num, num + len(eps)):
        pickle.dump(eps[i], open(dir + str(i) + '.dump', 'wb'))
    return


# Load episodes from directory. You'll have to which ones you want.
def load_episodes(dir, nums):
    eps = []
    for i in nums:
        eps.append(pickle.load(open(dir + str(i) + '.dump', 'rb')))
    return eps


# Plays back the episode
def playback(frames):
    for i in range(len(frames)):
        cv2.imshow('frame', repeat_upsample(frames[i][:, :, ::-1], 3, 3))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


# Prepares the data by splitting frames into observation and target
# Makes images black and white and rescales to be half of the original size
def forward_data(episodes, n_actions=4):
    stacks = []
    actions = []
    targets = []
    for j in range(len(episodes)):
        frames, inputs, _, _ = zip(*episodes[j])
        frames = list(frames)
        for i in range(len(episodes[j])):
            frames[i] = frames[i][::2, ::2]

        inputs = np.array(inputs)
        inputs[inputs == 1] = 0

        stack = []
        action = []
        target = []
        for i in range(len(episodes[j]) - 4):
            stack.append(np.concatenate(np.array(frames[i:i + 2]), axis=2))
            action.append(inputs[i + 1])
            target.append(np.concatenate(np.array(frames[i + 2:i + 4]), axis=2))

        stacks += stack
        actions += action
        targets += target

    return np.array(stacks), np.array(actions), np.array(targets)


def modal_data(episodes, n_actions=4):
    stacks = []
    actions = []
    targets = []
    for j in range(len(episodes)):
        frames, inputs, _, _ = zip(*episodes[j])
        frames = list(frames)
        for i in range(len(episodes[j])):
            frames[i] = frames[i][::2, ::2]

        inputs = np.array(inputs)
        inputs[inputs == 1] = 0

        stack = []
        action = []
        target = []
        for i in range(len(episodes[j]) - 6):
            stack.append(np.concatenate(np.array(frames[i:i + 4]), axis=2))
            action.append(inputs[i + 2])
            temp = np.copy(frames[i+4:i+6])
            temp[0] = temp[0] - frames[i+3]
            temp[1] = temp[1] - frames[i+4]
            target.append(np.concatenate(np.array(temp),axis = 2))

        stacks += stack
        actions += action
        targets += target

    return np.array(stacks), np.array(actions), np.array(targets)


def inverse_data(episodes, n_actions=4):
    stacks = []
    actions = []
    for j in range(len(episodes)):
        frames, inputs, _, _ = zip(*episodes[j])
        frames = list(frames)
        for i in range(len(episodes[j])):
            frames[i] = frames[i][::2, ::2]

        inputs = np.array(inputs)
        inputs[inputs == 1] = 0

        stack = []
        action = []
        for i in range(len(episodes[j]) - 3):
            stack.append(np.concatenate(np.array(frames[i:i + 4]), axis=2))
            action.append(inputs[i + 1])

        stacks += stack
        actions += action

    return np.array(stacks), np.array(actions)


# Data integrity tool. Shows that observations and target are logically constructed
def validate_data(d, a, t):
    for i in range(4):
        cv2.imshow('frame', repeat_upsample(np.array(d[:, :, 3*i:3*i+3]), 3, 3))
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break
    print(a)
    for i in range(2):
        cv2.imshow('frame', repeat_upsample(np.array(t[:, :, 3*i:3*i+3]), 3, 3))
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break
        print('close')
    print('done')
    if cv2.waitKey(3000) & 0xFF == ord('q'):
        return


def make_confusion_plot(i_model,f_model,c_model,random_episodes,human_episodes):

    print("Computing i_model confusion")

    fig, ax = plt.subplots(nrows = 2, ncols = 3, constrained_layout = True)

    # Make Confusion Matrix
    data,actions = inverse_data(random_episodes)
    r_score = np.argmax(i_model.predict([(data - np.mean(data, axis=0)) / 255.0]), axis=1)
    r_score[r_score == 1] = 0
    r_conf = confusion_matrix(actions, r_score)
    r_conf = normalize(r_conf,norm = 'l1')
    print("Random Score is " + str(np.trace(r_conf) / np.sum(r_conf)))
    print(r_conf)
    data, actions = inverse_data(human_episodes)
    h_score = np.argmax(i_model.predict([(data - np.mean(data, axis=0)) / 255.0]), axis=1)
    h_conf = confusion_matrix(actions, h_score)
    h_conf = normalize(h_conf, norm = 'l1')
    print("Human Score is " + str(np.trace(h_conf) / np.sum(h_conf)))
    print(h_conf)

    im = ax[0,0].imshow(r_conf)
    ax[0,0].set_xticks(np.arange(3))
    ax[0,0].set_yticks(np.arange(3))
    ax[0,0].set_xticklabels(['None', 'Right', 'Left'])
    ax[0,0].set_yticklabels(['None', 'Right', 'Left'])
    ax[0,0].set_title("Random Inverse")

    ax[1, 0].imshow(h_conf)
    ax[1, 0].set_xticks(np.arange(3))
    ax[1, 0].set_yticks(np.arange(3))
    ax[1, 0].set_xticklabels(['None', 'Right', 'Left'])
    ax[1, 0].set_yticklabels(['None', 'Right', 'Left'])
    ax[1, 0].set_title("Human Inverse")

    print("Computing f_model confusion")

    data, actions, targets = forward_data(random_episodes)
    sess = tf.Session()
    r_score = []
    pick = [0, 0, 0, 0]
    idx_actions = np.random.randint(0,len(actions), size = 100)
    actions = actions[idx_actions]
    for j in range(100):
        frames = (data[j] - np.mean(data, axis=0)) / 255.0
        frames = np.tile(frames, (4, 1, 1, 1))
        pick = np.sum(clipped_mse(f_model.predict([frames, np.arange(0, 4)])[:,:,:,3:6], targets[j][:,:,3:6]).eval(session=sess), axis=(1, 2))
        r_score.append(np.argmin(pick))
    r_score = np.array(r_score)
    r_score[r_score == 1] = 0
    actions[actions == 1] = 0
    r_conf = confusion_matrix(actions, r_score)
    r_conf = normalize(r_conf,norm = 'l1')
    print("Random Score is " + str(np.trace(r_conf) / np.sum(r_conf)))
    print(r_conf)

    data, actions, targets = forward_data(human_episodes)
    sess = tf.Session()
    h_score = []
    pick = [0, 0, 0, 0]
    idx_actions = np.random.randint(0, len(actions), size=100)
    actions = actions[idx_actions]
    for j in range(100):
        frames = (data[j] - np.mean(data, axis=0)) / 255.0
        frames = np.tile(frames, (4, 1, 1, 1))
        pick = np.sum(clipped_mse(f_model.predict([frames, np.arange(0, 4)])[:,:,:,3:6], targets[j][:,:,3:6]).eval(session=sess),axis=(1, 2))
        h_score.append(np.argmin(pick))
    h_score = np.array(h_score)
    h_score[h_score == 1] = 0
    actions[actions == 1] = 0
    h_conf = confusion_matrix(actions, h_score)
    h_conf = normalize(h_conf, norm = 'l1')
    print("Human Score is " + str(np.trace(h_conf) / np.sum(h_conf)))
    print(h_conf)

    ax[0, 1].imshow(r_conf)
    ax[0, 1].set_xticks(np.arange(3))
    ax[0, 1].set_yticks(np.arange(3))
    ax[0, 1].set_xticklabels(['None', 'Right', 'Left'])
    ax[0, 1].set_yticklabels(['None', 'Right', 'Left'])
    ax[0, 1].set_title("Random Forward")

    ax[1, 1].imshow(h_conf)
    ax[1, 1].set_xticks(np.arange(3))
    ax[1, 1].set_yticks(np.arange(3))
    ax[1, 1].set_xticklabels(['None', 'Right', 'Left'])
    ax[1, 1].set_yticklabels(['None', 'Right', 'Left'])
    ax[1, 1].set_title("Human Forward")

    print("Computing c_model confusion")

    # Make Confusion Matrix
    data, actions = inverse_data(random_episodes)
    data = data[:-2]
    actions = actions[2:]
    r_score = np.argmax(c_model.predict([(data - np.mean(data, axis=0)) / 255.0]), axis=1)
    r_score[r_score == 1] = 0
    r_conf = confusion_matrix(actions, r_score)
    r_conf = normalize(r_conf)
    print("Random Score is " + str(np.trace(r_conf) / np.sum(r_conf)))
    print(r_conf)
    data, actions = inverse_data(human_episodes)
    data = data[:-2]
    actions = actions[2:]
    h_score = np.argmax(c_model.predict([(data - np.mean(data, axis=0)) / 255.0]), axis=1)
    h_score[h_score == 1] = 0
    h_conf = confusion_matrix(actions, h_score)
    h_conf = normalize(h_conf,norm = 'l1')
    print("Human Score is " + str(np.trace(h_conf) / np.sum(h_conf)))
    print(h_conf)

    ax[0, 2].imshow(r_conf)
    ax[0, 2].set_xticks(np.arange(3))
    ax[0, 2].set_yticks(np.arange(3))
    ax[0, 2].set_xticklabels(['None', 'Right', 'Left'])
    ax[0, 2].set_yticklabels(['None', 'Right', 'Left'])
    ax[0, 2].set_title("Random Clone")

    ax[1, 2].imshow(h_conf)
    ax[1, 2].set_xticks(np.arange(3))
    ax[1, 2].set_yticks(np.arange(3))
    ax[1, 2].set_xticklabels(['None', 'Right', 'Left'])
    ax[1, 2].set_yticklabels(['None', 'Right', 'Left'])
    ax[1, 2].set_title("Human Clone")

    fig.colorbar(im, ax=list(ax.flatten()))
    plt.show()

    return