# Environment
import gym
from gym.utils.play import play
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
from keras.models import model_from_json

import util
import models


# Play Agent
def agent_play(env, model, mean, sup=[1, 1, 1, 1], temp=1.0):
    rew_total = 0
    for i in range(1):
        done = False
        lives = 5
        t_lives = 0
        count = 0
        frames = []
        while not done:
            if (abs(lives - t_lives) >= 1) or (count < 4):
                action = np.random.choice([0, 1, 2, 3], 1, p=[0.4, 0.2, 0.2, 0.2])
                frame, _, done, info = env.step(action)
                frames.append(frame[::2, ::2])
                if action == 1:
                    lives = t_lives
                    t_lives = info['ale.lives']
            else:
                dist = np.power(model.predict([(np.array([np.concatenate(frames[-4:], axis=2)]) - mean) / 255.0])[0],
                                temp)
                dist = dist * np.array(sup)
                dist /= np.sum(dist)
                action = np.random.choice([0, 1, 2, 3], 1, p=dist)[0]
                frame, rew, done, info = env.step(action)
                rew_total += rew
                frames.append(frame[::2, ::2])
                t_lives = info['ale.lives']
            cv2.imshow('frame', util.repeat_upsample(frames[count][:, :, ::-1], 6, 6))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            count += 1
    env.close()
    return rew_total


# Random policy evaluation
def random_play(env):
    rew_total = 0
    episodes = []
    for i in range(1):
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
                rew_total += episode[-1][1]
        episodes.append(episode)
    env.close()
    return rew_total


if __name__ == '__main__':

    env = util.make_environment('BreakoutNoFrameskip-v4')

    episodes, n_actions = util.record_episode(env, num=1)
    data, actions = util.inverse_data(episodes, n_actions)
    mu = np.mean(data, axis=0)

    # Optionally load json and create model
    load_model = True
    if load_model is True:
        json_file = open('Production_Models/final_i_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        i_model = model_from_json(loaded_model_json)
        # load weights into new model
        i_model.load_weights("Production_Models/final_i_model.h5")
        print("Loaded model from disk")

        # View data and target to verify integrity
        # validate_data(data[4], actions[4:])'''

    # Optionally load json and create model
    load_model = True
    if load_model is True:
        json_file = open('Production_Models/f_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        f_model = model_from_json(loaded_model_json)
        # load weights into new model
        f_model.load_weights("Production_Models/f_model.h5")
        print("Loaded model from disk")

        # View data and target to verify integrity
        # validate_data(data[4], actions[4:])'''

    # Optionally load json and create model
    load_model = True
    if load_model is True:
        json_file = open('Production_Models/c_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        c_model = model_from_json(loaded_model_json)
        # load weights into new model
        c_model.load_weights("Production_Models/c_model.h5")
        print("Loaded model from disk")

    rew = []
    for i in range(1):
        env = gym.make("BreakoutNoFrameskip-v4")
        env = gym.wrappers.Monitor(env, 'Test_Recording', force=True)
        env = util.MaxAndSkipEnv(env, 2)
        env.reset()
        temp = agent_play(env, c_model, mu, sup=[1, 0, 2, 1])
        rew.append(temp)
        print(np.mean(rew))
    env.close()
    print("Mean Reward: " + str(np.mean(rew)))
    print("Median Reward: " + str(np.median(rew)))