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
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def train_inverse(i_model, model_path, data_path, bco_episodes = [], epoch=30, data_size=30, frame_num = 4, action_num=2, use_images = False, verbose = True):

    episodes = util.load_episodes(data_path, range(0,0 + data_size))+bco_episodes
    data, actions = util.inverse_vector_data(episodes, frame_num = frame_num, action_num = action_num, use_images = use_images)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(list(actions)), list(actions))
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=False, patience=500)
    mc = ModelCheckpoint('i_model.h5', monitor='val_acc', mode='max', verbose=False, save_best_only=True)
    i_model.fit([data], [to_categorical(actions, action_num)], validation_split=0.3, batch_size=64, epochs=epoch, shuffle=True,
                class_weight=class_weights, verbose = verbose, callbacks=[es, mc])

    # serialize model to JSON
    model_json = i_model.to_json()
    with open(model_path + "i_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #i_model.save_weights(model_path + "i_model.h5")

    i_model = load_model('i_model.h5')

    return i_model


def train_clone(c_model, i_model, model_path, data_path, bco_episodes = [], epoch = 30, data_size=30, frame_num = 4, action_num=2, use_images = False, verbose = True):

    episodes = util.load_episodes(data_path, range(0, 0 + data_size))+bco_episodes
    data, actions = util.linear_vector_data(episodes,frame_num = frame_num, action_num = action_num, use_images = use_images)
    actions = np.argmax(i_model.predict([data]),axis = 1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(list(actions)), list(actions))
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=False, patience=500)
    mc = ModelCheckpoint('c_model.h5', monitor='val_acc', mode='max', verbose=False, save_best_only=True)
    c_model.fit([data[:-frame_num+1]], to_categorical(actions[frame_num-2:-1], action_num), validation_split=0.3, batch_size=64, epochs=epoch, shuffle=True, verbose = verbose,
                class_weight=class_weights, callbacks=[es, mc])

    # serialize model to JSON
    model_json = c_model.to_json()
    with open(model_path + "c_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #c_model.save_weights(model_path + "c_model.h5")

    c_model = load_model('c_model.h5')

    return c_model


def train_modal(m_model, model_path, data_path, epoch=30, data_size=30, frame_num = 4, latent_action_num=2, use_images = False):

    print("Training Modal Model")
    for k in range(epoch):
        print("Epoch " + str(k))
        for i in range(0, data_size, 30):
            episodes = util.load_episodes(data_path, range(i, i + 30))
            data, actions, targets = util.vector_modal_data(episodes, frame_num = frame_num, latent_action_num = latent_action_num)
            m_model.fit([data], [np.moveaxis(np.repeat(np.array([targets]), latent_action_num, axis=0), 0, 1)], batch_size=64,
                        epochs=1, shuffle=True, verbose=True)
            del data
            del episodes
            del actions

    # serialize model to JSON
    model_json = m_model.to_json()
    with open(model_path + "m_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    m_model.save_weights(model_path + "m_model.h5")

    return m_model


def train_latent(l_model, m_model, model_path, data_path, epoch=30, data_size=30, frame_num = 4, latent_action_num=2, use_images = False):

    print("Training Modal Model")
    for k in range(epoch):
        print("Epoch " + str(k))
        for i in range(0, data_size, 30):
            episodes = util.load_episodes(data_path, range(i, i + 30))
            data, actions, targets = util.vector_modal_data(episodes, frame_num = frame_num)
            pred_image = m_model.predict([data])
            #print(data.shape)
            #print(pred_image.shape)
            #print(np.moveaxis(np.repeat(np.array([targets]), latent_action_num, axis=0), 0,1).shape)
            latent_actions = models.vector_argmin_mse(pred_image,np.moveaxis(np.repeat(np.array([targets]), latent_action_num, axis=0), 0,1))
            #print(latent_actions.shape)
            #print(to_categorical(latent_actions, latent_action_num).shape)
            class_weights = class_weight.compute_class_weight('balanced', np.unique(latent_actions), latent_actions)
            l_model.fit([data], [to_categorical(latent_actions, latent_action_num)], class_weight=class_weights,
                        batch_size=16, epochs=1, shuffle=True, verbose=True)
            del data
            del episodes
            del actions

    # serialize model to JSON
    model_json = l_model.to_json()
    with open(model_path + "l_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    l_model.save_weights(model_path + "l_model.h5")

    return l_model


def train_action(a_model, l_model, episodes, model_path, env, epoch = 30, frame_num = 4, latent_action_num = 2, action_num = 2, use_images = False):

    for i in range(epoch):
        if i == 0:
            reward, new_episodes = model_evaluate(l_model, env, num_episodes=2, frame_num=frame_num, action_num=action_num)
            print(reward)
            episodes = episodes + new_episodes.copy()
            data, actions, targets = util.vector_modal_data(episodes, frame_num=frame_num)
        else:
            reward, new_episodes = latent_model_evaluate(l_model, a_model, env, num_episodes=2, frame_num=frame_num, action_num=action_num)
            print(reward)
            episodes = episodes+new_episodes.copy()
            data, actions, targets = util.vector_modal_data(episodes, frame_num = frame_num)
        # Actual
        pred_image = m_model.predict([data])
        latent_actions = models.vector_argmin_mse(pred_image, np.moveaxis(np.repeat(np.array([targets]), latent_action_num, axis=0), 0, 1))

        # Predicted
        predicted_latent_actions = np.argmax(l_model.predict([data]), axis=1)

        # Get indices of matches between prediction and observation
        matches = np.argwhere(latent_actions == predicted_latent_actions).flatten()
        class_weights = class_weight.compute_class_weight('balanced', np.unique(list(actions[matches])),
                                                          list(actions[matches]))
        a_model.fit([data[matches], to_categorical(latent_actions[matches], latent_action_num)],
                    to_categorical(actions[matches], action_num), class_weight=class_weights, batch_size=16, epochs=5,
                    shuffle=True, verbose=True)

    # serialize model to JSON
    model_json = a_model.to_json()
    with open(model_path + "a_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    a_model.save_weights(model_path + "a_model.h5")

    return a_model


# Shrink image from (400,600,3) -> (105,80,3)
def wrap_image(img):
    return np.resize(np.transpose(img[::5,::6],(1,0,2)),(105,80,3))


# Push-Cart Evaluation
def model_evaluate(model, env, num_episodes = 30, frame_num = 4, action_num = 2):
    """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_episodes: (int) number of episodes to evaluate it
        :return: (float) Mean reward for the last num_episodes and episode data
    """

    # This function will only work for a single Environment
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
            action = np.random.choice(range(action_num), 1, p=dist)[0]

            obs, reward, done, info = env.step(action)
            #obs = wrap_image(env.render(mode='rgb_array'))
            #img = env.render(mode='rgb_array')
            frames.append(obs)
            episode_rewards.append(reward)
            episode[-1].insert(1,action)
            episode.append([obs, reward, done, info])
            count += 1

        env.close()
        #print(sum(episode_rewards))
        all_episode_rewards.append(sum(episode_rewards))
        episodes.append(episode)

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    print("Reward Standard Deviation:", np.sqrt(np.var(all_episode_rewards)), "Num episodes:", num_episodes)

    return mean_episode_reward, episodes


# Push-Cart Evaluation
def latent_model_evaluate(model, a_model, env, num_episodes = 30, frame_num = 4, action_num = 2):
    """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_episodes: (int) number of episodes to evaluate it
        :return: (float) Mean reward for the last num_episodes and episode data
    """

    # This function will only work for a single Environment
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

            raw = model.predict([[np.concatenate(np.array(frames[count:count+frame_num]))]])
            latent_action = np.argmax(raw, axis=1)

            dist = a_model.predict([[np.concatenate(np.array(frames[count:count+frame_num]))],to_categorical(latent_action, latent_action_num)])[0]
            dist /= np.sum(dist)
            action = np.random.choice(range(action_num), 1, p=dist)[0]

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
    print("Reward Standard Deviation:", np.sqrt(np.var(all_episode_rewards)), "Num episodes:", num_episodes)

    return mean_episode_reward, episodes


if __name__ == '__main__':

    # Vector BCO
    env = gym.make('CartPole-v0')
    learning_rates = [0.001, 0.001]
    dim = 4
    frame_num = 4
    action_num = 2

    # Semi-Supervised BC
    pre_train_data = 1
    post_train_data = 10
    i_model = models.linear_inverse_model(learning_rate=learning_rates[0], dim=dim, frame_num=frame_num,
                                          action_num=action_num)
    c_model = models.linear_clone_model(learning_rate=learning_rates[1], dim=dim, frame_num=frame_num,
                                        action_num=action_num)

    # Use a small collection of samples to train inverse model
    i_model = train_inverse(i_model, "./Basic_Models/", "./Vector_Rollouts/", data_size=pre_train_data,
                            epoch=500, frame_num=frame_num, action_num=action_num, use_images=False, verbose=False)
    # Use inverse model to label data and then train policy
    c_model = train_clone(c_model, i_model, "./Basic_Models/", "./Vector_Rollouts/", epoch=500, frame_num=frame_num,
                          action_num=action_num, use_images=False, verbose=False)

    rewards = []
    episodes = []
    if post_train_data > 0:
        for i in range(post_train_data):
            reward, new_episodes = model_evaluate(c_model, env, num_episodes=100, frame_num=frame_num,
                                                  action_num=action_num)
            rewards.append(reward)
            episodes += new_episodes[:1]
            i_model = train_inverse(i_model, "./Basic_Models/", "./Vector_Rollouts/", data_size=pre_train_data,
                                    bco_episodes=episodes, epoch=500,
                                    frame_num=frame_num, action_num=action_num, use_images=False, verbose=False)
            c_model = train_clone(c_model, i_model, "./Basic_Models/", "./Vector_Rollouts/", epoch=500,
                                  frame_num=frame_num, action_num=action_num, use_images=False, verbose=False)
            #reward, new_episodes = model_evaluate(c_model, env, num_episodes=100, frame_num=frame_num,
            #                                      action_num=action_num)
            #rewards.append(reward)
    else:
        reward, new_episodes = model_evaluate(c_model, env, num_episodes=100, frame_num=frame_num,
                                              action_num=action_num)
        rewards.append(reward)
    print(rewards)

    # Train BCO using a pre/post demonstration split
    '''pre_train_data = 7
    post_train_data = 8
    i_model = models.linear_inverse_model(learning_rate=learning_rates[0], dim = dim, frame_num = frame_num, action_num = action_num)
    c_model = models.linear_clone_model(learning_rate=learning_rates[1], dim = dim, frame_num = frame_num, action_num = action_num)
    i_model = train_inverse(i_model,"./Basic_Models/", "./Inverse_Vector_Rollouts/", data_size = pre_train_data, epoch = 5000, frame_num = frame_num, action_num = action_num, use_images = False, verbose = False)
    c_model = train_clone(c_model, i_model, "./Basic_Models/","./Vector_Rollouts/", epoch = 5000, frame_num = frame_num, action_num = action_num, use_images = False, verbose = False)

    rewards = []
    if post_train_data > 0:
        reward, new_episodes = model_evaluate(c_model, env, num_episodes=100, frame_num = frame_num, action_num = action_num)
        rewards.append(reward)
        episodes = new_episodes[:post_train_data]
        i_model = train_inverse(i_model, "./Basic_Models/", "./Inverse_Vector_Rollouts/", data_size=pre_train_data, bco_episodes = episodes, epoch=5000,
                                frame_num=frame_num, action_num=action_num, use_images=False, verbose = False)
        c_model = train_clone(c_model, i_model, "./Basic_Models/", "./Vector_Rollouts/", epoch=5000,
                              frame_num=frame_num, action_num=action_num, use_images=False, verbose = False)
        reward, new_episodes = model_evaluate(c_model, env, num_episodes=100, frame_num=frame_num, action_num=action_num)
        rewards.append(reward)
    else:
        reward, new_episodes = model_evaluate(c_model, env, num_episodes=100, frame_num=frame_num, action_num=action_num)
        rewards.append(reward)
    print(rewards)'''

    # Vector LIL
    '''env = gym.make('CartPole-v0')
    learning_rates = [0.001, 0.001] # serialize model to JSON
    dim = 4
    frame_num = 4
    latent_action_num = 2
    action_num = 2
    m_model = models.vector_modal_model(dim = dim, frame_num = frame_num, latent_action_num = latent_action_num)
    l_model = models.vector_latent_model(dim = dim, frame_num = frame_num, latent_action_num = latent_action_num)
    a_model = models.vector_action_model(dim = dim, frame_num = frame_num, latent_action_num = latent_action_num, action_num = action_num)

    m_model = train_modal(m_model, "./Car_Models/", "./Vector_Rollouts/", epoch=30, frame_num=frame_num, latent_action_num=latent_action_num, use_images=False)
    l_model = train_latent(l_model, m_model, "./Car_Models/", "./Vector_Rollouts/", epoch = 30, frame_num=frame_num, latent_action_num= latent_action_num)
    reward, episodes = model_evaluate(l_model, env, num_episodes=100, frame_num=frame_num, action_num=action_num)
    #a_model = train_action(a_model, l_model, [], "./Car_Models/", env, epoch = 100, frame_num = frame_num, latent_action_num = latent_action_num, action_num = action_num)'''
    #reward, episodes = latent_model_evaluate(l_model,a_model, env, num_episodes=100, frame_num=frame_num, action_num=action_num)
