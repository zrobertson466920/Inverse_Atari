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
from keras.utils.np_utils import to_categorical
from keras.backend import set_value

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


# Play Partial Agent
def latent_play(env, latent_model, action_model, mean):
    rew_total = 0
    episodes = []
    p_dict = {0:3,1:0,2:2,3:3}
    for i in range(1):
        done = False
        lives = 5
        t_lives = 0
        count = 0
        frames = []
        episode = []
        episode.append([env.reset(), 0, done, None])
        while not done:
            if (abs(lives - t_lives) >= 1) or (count < 4):
                action = np.random.choice([0, 1, 2, 3], 1, p=[0.4, 0.2, 0.2, 0.2])
                frame, _, done, info = env.step(action)
                frames.append(frame[::2, ::2])
                episode[-1].insert(1, action)
                #episode.append(list(env.step(action)))
                if action == 1:
                    lives = t_lives
                    t_lives = info['ale.lives']
            else:
                latent_action = np.zeros((4,))
                latent_action[np.argmax(latent_model.predict([np.array([np.concatenate(frames[-4:], axis=2)])]),axis = 1)[0]] = 1
                '''dist = action_model.predict([np.array([np.concatenate(frames[-4:], axis=2)]),np.array([latent_action])])[0]
                dist /= np.sum(dist)
                action = np.random.choice([0, 1, 2, 3], 1, p=dist)[0]'''
                action = np.argmax(action_model.predict([np.array([np.concatenate(frames[-4:], axis=2)]), np.array([latent_action])])[0])
                if np.random.randint(0,10,1) == 0:
                    if np.random.randint(0,2,1) == 0:
                        action = 2
                    else:
                        action = 3
                #action = p_dict[np.argmax(latent_model.predict([np.array([np.concatenate(frames[-4:], axis=2)])]), axis=1)[0]]
                frame, rew, done, info = env.step(action)
                rew_total += rew
                frames.append(frame[::2, ::2])
                t_lives = info['ale.lives']
                episode[-1].insert(1, action)
                #episode.append(list(env.step(action)))
            cv2.imshow('frame', util.repeat_upsample(frames[count][:, :, ::-1], 6, 6))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            count += 1
        episodes.append(episode)
    env.close()
    return episodes, rew_total


def predict(input):
    return latent_model.predict(input)


def production_run(env):
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
        # validate_data(data[4], actions[4:])

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
        # validate_data(data[4], actions[4:])

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


def test(env):
    l_model = models.latent_model(learning_rate=0.001)
    m_model = models.modal_model(learning_rate=0.001)
    for k in range(1):
        for i in range(3):
            episodes, n_actions = util.record_episode(env, num=2)
            # episodes = util.load_episodes("/content/gdrive/My Drive/Colab Notebooks/Trained_Model/",
            #                              list(range(2 * i, 2 * i + 2)))
            data, actions, targets = util.modal_data(episodes)
            m_model.fit([data], [np.moveaxis(np.repeat(np.array([targets]),4,axis = 0),0,1)], batch_size=16, epochs=1, validation_split=0.2, shuffle=True)
            pred_image = m_model.predict([data])
            latent_actions = models.argmin_mse(pred_image,np.moveaxis(np.repeat(np.array([targets]), 4, axis=0), 0, 1))
            l_model.fit([data], [to_categorical(latent_actions)], class_weight = 'auto', batch_size=16, epochs=1, validation_split=0.2, shuffle=True)
            print(pred_image.shape)
            print(targets.shape)
            print(np.mean(np.square(pred_image - np.moveaxis(np.repeat(np.array([targets]), 4, axis=0), 0, 1)),(2, 3, 4), keepdims=True)[:, :, 0, 0, 0])
            #for i in range(4):
            #    plt.imshow(np.ndarray.astype(pred_image[5][i][:, :, 3:6], dtype='uint8'))
            #    plt.show()
            #new_image, _, action = predict([(data - np.mean(data, axis=0)) / 255])
            #temp = np.moveaxis(np.repeat(np.array([targets]), 4, axis=0), 0, 1)
            #latent_actions = models.argmin_mse(new_image,temp)

            #action_model.fit([(data - np.mean(data, axis=0)) / 255,to_categorical(latent_actions,num_classes=4)],to_categorical(actions,num_classes=4), batch_size=16, epochs=1, validation_split=0.2, shuffle=True)
            #latent_model.fit([(data - np.mean(data, axis=0)) / 255],targets, batch_size=16, epochs=1,validation_split=0.2, shuffle=True)
            # f_model.fit([(data-np.mean(data,axis=0))/255,actions],[targets],batch_size = 64, epochs = 10, validation_split = 0.2, shuffle = True)


if __name__ == '__main__':

    env = util.make_environment('BreakoutNoFrameskip-v4')

    #test(env)

    #episodes, n_actions = util.record_episode(env, num=1)
    #data, actions, targets = util.modal_data(episodes, 4)
    #print(actions[10:])
    #util.validate_data(data[10],actions[10],targets[10])

    # Optionally load json and create model
    load_model = True
    if load_model is True:
        json_file = open('Test_Models/a_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        action_model = model_from_json(loaded_model_json)
        # load weights into new model
        action_model.load_weights("Test_Models/a_model.h5")
        print("Loaded model from disk")

    # Optionally load json and create model
    load_model = True
    if load_model is True:
        json_file = open('Test_Models/final_l_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        latent_model = model_from_json(loaded_model_json)
        # load weights into new model
        latent_model.load_weights("Test_Models/final_l_model.h5")
        print("Loaded model from disk")
    else:
        latent_model = models.latent_model(learning_rate=0.0001)

    rew = []
    #episodes, n_actions = util.record_episode(env, num=1)
    episodes = util.load_episodes("Human_Model/", [1])
    data, actions, targets = util.modal_data(episodes, 4)
    '''data = data[:]
    actions = actions[0:]
    h_score = np.argmax(action_model.predict([(data - np.mean(data, axis=0)) / 255.0]), axis=1)
    h_score[h_score == 1] = 0
    h_conf = confusion_matrix(actions, h_score)
    #h_conf = normalize(h_conf, norm='l1')
    print("Human Score is " + str(np.trace(h_conf) / np.sum(h_conf)))
    print(h_conf)
    print(h_score[150:450])
    print(actions[150:450])'''
    mu = np.mean(data, axis=0)
    for i in range(100):
        env = gym.make("BreakoutNoFrameskip-v4")
        env = util.MaxAndSkipEnv(env, 2)
        env.seed(0)
        env.reset()
        # temp = random_play(env)
        _, temp = latent_play(env, latent_model,action_model, mu)
        rew.append(temp)
        print(temp)
    print("Mean Reward: " + str(np.mean(rew)))
    print("Std: " + str(np.std(rew)))