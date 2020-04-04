# Environment
import gym
#from gym.utils.play import play
#from gym.spaces import prng

# Utility
import cv2
import numpy as np
from collections import deque
import argparse
from functools import partial
import pickle
#%tensorflow_version 1.13.1
import tensorflow as tf
import matplotlib.pyplot as plt

# Models
import models
import util

# Sklearn
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

# Keras
from keras import Sequential, Model, Input
from keras.utils import to_categorical
from keras.layers import Dense, Flatten,multiply, Dropout, Reshape, Activation, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras import metrics
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session


def train_inverse(i_model, data_path, epoch=30, data_size=30, frame_num = 4, action_num=2, use_images = False):

    print("Training Inverse Model")
    for k in range(epoch):
        print("Epoch " + str(k))
        for i in range(0, data_size, 30):
            episodes = util.load_episodes(data_path, range(i, i + 30))
            data, actions = util.inverse_vector_data(episodes, frame_num = frame_num, action_num = action_num, use_images = use_images)
            class_weights = class_weight.compute_class_weight('balanced', np.unique(list(actions)), list(actions))
            i_model.fit([data], [to_categorical(actions, action_num)], batch_size=64, epochs=5, shuffle=True,
                        class_weight=class_weights)
            del data
            del episodes
            del actions

    return i_model


def train_clone(c_model, i_model, data_path, epoch = 30, data_size=30, frame_num = 4, action_num=2, use_images = False):

    print("Training Clone Model")
    for k in range(epoch):
        print("Epoch " + str(k))
        for i in range(0, data_size, 30):
            episodes = util.load_episodes(data_path, range(i, i + 30))
            data, actions = util.linear_vector_data(episodes,frame_num = frame_num, action_num = action_num, use_images = use_images)
            actions = np.argmax(i_model.predict([data]),axis = 1)
            class_weights = class_weight.compute_class_weight('balanced', np.unique(list(actions)), list(actions))
            c_model.fit([data[:-frame_num+1]], to_categorical(actions[frame_num-2:-1], action_num), batch_size=64, epochs=5, shuffle=True, verbose = True, validation_split = 0.2,
                        class_weight=class_weights)
            del episodes
            del data
            del actions

    return c_model


# Shrink image from (400,600,3) -> (105,80,3)
def wrap_image(img):
    return np.resize(np.transpose(img[::5,::6],(1,0,2)),(105,80,3))


# Push-Cart Evaluation
def model_evaluate(model, num_episodes = 30, frame_num = 4, action_num = 2):
    """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_episodes: (int) number of episodes to evaluate it
        :return: (float) Mean reward for the last num_episodes and episode data
    """

    # This function will only work for a single Environment
    env = gym.make('CartPole-v1')
    all_episode_rewards = []
    episodes = []
    for i in range(num_episodes):
        episode_rewards = []
        episode = []
        done = False
        obs = env.reset()
        episode.append([obs, 0, done, None])
        count = 0
        frames = []
        for f in range(frame_num):
            frames.append(obs)
        while not done:

            dist = model.predict([[np.concatenate(np.array(frames[count:count+frame_num]))]])[0]
            dist /= np.sum(dist)
            action = np.random.choice([0, 1], 1, p=dist)[0]

            obs, reward, done, info = env.step(action)
            #obs = wrap_image(env.render(mode='rgb_array'))
            frames.append(obs)
            episode_rewards.append(reward)
            episode[-1].insert(1,action)
            episode.append([obs, reward, done, info])
            count += 1

        env.close()
        print(sum(episode_rewards))
        all_episode_rewards.append(sum(episode_rewards))
        episodes.append(episode)

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward, episodes


if __name__ == '__main__':

    learning_rates = [0.001, 0.001]
    frame_num = 4
    action_num = 2
    i_model = models.inverse_model(learning_rate=learning_rates[0], frame_num = frame_num, action_num = action_num)
    c_model = models.clone_model(learning_rate=learning_rates[1], frame_num = frame_num, action_num = action_num)
    i_model = train_inverse(i_model,"./Inverse_Rollouts/", epoch = 50, frame_num = frame_num, action_num = action_num, use_images = True)
    c_model = train_clone(c_model, i_model, "./Rollouts/", epoch = 50, frame_num = frame_num, action_num = action_num, use_images = True)
    model_evaluate(c_model, num_episodes=30, frame_num = frame_num, action_num = action_num)
