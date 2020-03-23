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
from sklearn.utils import class_weight
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import tensorflow as tf

import hdbscan
import matplotlib.cm as cm

from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.backend import set_value
from keras import backend as K
from keras.models import Model

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
        episode = []
        episode.append([env.reset(), 0, done, None])
        while not done:
            if (abs(lives - t_lives) >= 1) or (count < 4):
                action = np.random.choice([0, 1, 2, 3], 1, p=[0.4, 0.2, 0.2, 0.2])
                frame, rew, done, info = env.step(action)
                episode[-1].insert(1, action)
                episode.append(list((frame, rew, done, info)))
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
                episode[-1].insert(1, action)
                episode.append(list((frame, rew, done, info)))
                rew_total += rew
                frames.append(frame[::2, ::2])
                t_lives = info['ale.lives']
            '''cv2.imshow('frame', util.repeat_upsample(frames[count][:, :, ::-1], 6, 6))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break'''
            count += 1
    env.close()
    return episode, rew_total


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
    return episodes, rew_total


# Play Partial Agent
def latent_play(env, latent_model, action_model, l_num=4, guess=None, show=False):
    rew_total = 0
    episodes = []

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
                frame, rew, done, info = env.step(action)
                frames.append(frame[::2, ::2])
                episode[-1].insert(1, action)
                episode.append(list((frame, rew, done, info)))
                if action == 1:
                    lives = t_lives
                    t_lives = info['ale.lives']
            else:
                if guess != None:
                    action = guess[
                        np.argmax(latent_model.predict([np.array([np.concatenate(frames[-4:], axis=2)])]), axis=1)[0]]
                else:
                    latent_action = np.zeros((l_num,))
                    latent_action[np.argmax(latent_model.predict([np.array([np.concatenate(frames[-4:], axis=2)])]), axis=1)[0]] = 1
                    print(latent_action)
                    dist = \
                    action_model.predict([np.array([np.concatenate(frames[-4:], axis=2)]), np.array([latent_action])])[
                        0]
                    #dist = pow(dist,0.5)
                    dist = dist + 0.1
                    dist /= np.sum(dist)
                    action = np.random.choice([0, 1, 2, 3], 1, p=dist).flatten()[0]
                frame, rew, done, info = env.step(action)
                rew_total += episode[-1][1]
                frames.append(frame[::2, ::2])
                t_lives = info['ale.lives']
                episode[-1].insert(1, action)
                episode.append(list((frame, rew, done, info)))
            if show is True:
                cv2.imshow('frame', util.repeat_upsample(frames[count][:, :, ::-1], 6, 6))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            count += 1
        episodes.append(episode)
    env.close()
    return episodes, rew_total


def predict(input):
    return latent_model.predict(input)


def production_run(env,mu):
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


# Plays back the episode
def playback(frames, zoom = 3):
    for i in range(len(frames)):
        cv2.imshow('frame', repeat_upsample(frames[i][:,:,::-1], zoom, zoom))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


def drawArrow(A, B, i):
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=0.005, length_includes_head=True, color = 'C' + str(i))


def visualize_trajectories(m_model,l_model,a_model,start,stop):
    raw_embedding = []
    raw_actions = []
    real_actions = []
    mu = pickle.load(open('Production_Models/' + 'mu' + '.dump', 'rb'))
    for j in range(start, stop):
        #episode = util.load_episodes("Random_Model/", [j])
        #episode = util.load_episodes("Human_Model/", [j])
        env = gym.make("BreakoutNoFrameskip-v4")
        env = util.MaxAndSkipEnv(env, 2)
        env.seed(j)
        env.reset()
        #episode, rew = latent_play(env, l_model,a_model, guess = {0:2,1:2,2:3,3:0}, show = False)
        episode, temp = agent_play(env, a_model, mean=mu)
        temp = np.array(temp)
        episode = np.array([episode])
        #episode = np.array([episode])
        #print(rew)
        #data, actions, targets = util.inverse_data(episode)
        data, actions = util.inverse_data(episode)

        layer_name = 'dense_28'
        intermediate_layer_model = Model(inputs=m_model.input,
                                         outputs=m_model.get_layer(layer_name).output)

        for i in range(len(data)):
            #state_embedding = np.ravel(intermediate_layer_model.predict(np.array([data[i]])))
            state_embedding = intermediate_layer_model.predict(np.array([data[i]]))[0]
            #latent_action = np.ravel(l_model.predict(np.array([data[i]])))
            #raw_embedding.append(np.concatenate((state_embedding,latent_action)))
            raw_embedding.append(state_embedding)
            #raw_actions.append(latent_action)
            real_actions.append(actions[i])

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale

    pca = PCA(n_components=2)
    #raw_embedding = scale(raw_embedding,axis = 0)
    #raw_actions = scale(raw_actions,axis = 0)
    #raw_encoding = [np.concatenate((a,b)) for a,b in zip(raw_embedding,raw_actions)]
    #raw_encoding = scale(raw_encoding,axis = 0)
    new_encoding = np.flip(pca.fit_transform(raw_embedding))

    print(new_encoding.shape)
    for i in range(new_encoding.shape[0]):
        if i % 100 in list(range(0, 10)):
            try:
                drawArrow(new_encoding[i], new_encoding[i + 1], int(real_actions[i]))
            except:
                print(new_encoding[i].shape)
    #plt.scatter(new_encoding[:,0],new_encoding[:,1])
    plt.xlim(np.min(-3000), np.max(3000))
    plt.ylim(np.min(-3000), np.max(3000))
    plt.title('Inverse Trajectory')
    plt.savefig('inverse_path.png')
    plt.show()


def collapse_trajectories(m_model,l_model,a_model,start,stop):
    raw_embedding = []
    raw_actions = []
    for j in range(start, stop):
        #episode = util.load_episodes("Human_Model/", [j])
        #episode,_ = random_play(util.make_environment('BreakoutNoFrameskip-v4'))
        env = gym.make("BreakoutNoFrameskip-v4")
        env = util.MaxAndSkipEnv(env, 2)
        env.seed(j)
        env.reset()
        #episode, rew = latent_play(env, l_model, a_model, guess={0: 2, 1: 2, 2: 3, 3: 0})
        data, actions, targets = util.modal_data(episode)
        #frames, inputs, _, _ = zip(*episode[0])
        #playback(frames)

        layer_name = 'dense_28'
        intermediate_layer_model = Model(inputs=m_model.input,
                                         outputs=m_model.get_layer(layer_name).output)

        for i in range(len(data)):
            raw_embedding.append(intermediate_layer_model.predict(np.array([data[i]]))[0])
            raw_actions.append(actions[i])

    from sklearn.decomposition import PCA
    import seaborn as sns

    pca = PCA(n_components=2)
    new_encoding = np.flip(pca.fit_transform(raw_embedding))

    # Use HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
    cluster_labels = clusterer.fit_predict(new_encoding)
    #clusterer.condensed_tree_.plot()
    #plt.show()
    print(cluster_labels)

    colors = cm.Spectral(np.linspace(0, 1, len(cluster_labels)))
    nodes = []

    for i in np.unique(cluster_labels):

        if i != -1:
            class_member_mask = (cluster_labels == i)

            nodes.append(np.mean(new_encoding[class_member_mask], axis=0))


    '''for i, c in zip(cluster_labels,colors):

        if i != -1:
            class_member_mask = (cluster_labels == i)

            nodes.append(np.mean(new_encoding[class_member_mask],axis = 0))

            #plt.scatter(np.mean(new_encoding[class_member_mask][:,0],0),np.mean(new_encoding[class_member_mask][:,1],axis = 0),color = c, s = 15)
            plt.scatter(new_encoding[class_member_mask][:, 0],
                        new_encoding[class_member_mask][:, 1], color=c, s=15)'''

    for i in range(new_encoding.shape[0]):
        try:
            if cluster_labels[i] != -1 and cluster_labels[i+1] != -1:
                drawArrow(nodes[cluster_labels[i]], nodes[cluster_labels[i + 1]], raw_actions[cluster_labels[i]])
            #else:
                #drawArrow(new_encoding[i], new_encoding[i + 1], 1)
        except:
            print(nodes[cluster_labels[i]])

    # plt.scatter(new_encoding[:,0],new_encoding[:,1])
    plt.xlim(np.min(-3000), np.max(3000))
    plt.ylim(np.min(-3000), np.max(3000))
    plt.show()


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
        json_file = open('Production_Models/m_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        m_model = model_from_json(loaded_model_json)
        # load weights into new model
        m_model.load_weights("Production_Models/m_model.h5")
        print("Loaded model from disk")

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

    # Optionally load json and create model
    load_model = True
    if load_model is True:
        '''json_file = open('Production_Models/l_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        latent_model = model_from_json(loaded_model_json)'''
        latent_model = models.latent_model(learning_rate=0.001)
        # load weights into new model
        latent_model.load_weights("Production_Models/l_model.h5")
        print("Loaded model from disk")
    else:
        latent_model = models.latent_model(learning_rate=0.0001)

    visualize_trajectories(m_model,latent_model,c_model,0,10)

    '''a_model = models.alt_action_model(latent_model,learning_rate=0.001)
    l_num = 4
    rew = []
    for k in range(1):
        print(k)
        for i in range(0, 30, 1):
            # episodes, n_actions = util.record_episode(env,num = 5)
            # episodes = util.load_episodes("/content/gdrive/My Drive/Colab Notebooks/Trained_Model/", list(range(5 * 1, 5 * 1 + 5)))
            # episodes = util.load_episodes("/content/gdrive/My Drive/Colab Notebooks/Human_Model/",[i])
            env = gym.make("BreakoutNoFrameskip-v4")
            env = util.MaxAndSkipEnv(env, 2)
            env.seed(i)
            env.reset()
            if i < 3:
                #episodes, temp = latent_play(env, latent_model, a_model, guess={0:2,1:2,2:3,3:0}, show = True)
                episodes, temp = random_play(env)
            else:
                episodes, temp = latent_play(env, latent_model, a_model, show = True)
            # episodes, temp = latent_play(env,l_model,a_model)
            # print(temp)
            data, actions, targets = util.modal_data(episodes)
            pred_image = m_model.predict([data])
            latent_actions = models.argmin_mse(pred_image,np.moveaxis(np.repeat(np.array([targets]),4,axis = 0),0,1))
            #latent_actions = np.argmax(latent_model.predict([data]), axis=1)
            actions = actions.astype(dtype='int')
            class_weights = class_weight.compute_class_weight('balanced', np.unique(list(actions)), list(actions))
            a_model.fit([data, to_categorical(latent_actions, l_num)], to_categorical(actions, 4),
                        class_weight=class_weights, batch_size=16, epochs=5, shuffle=True, verbose=True)
            print(temp)
            rew.append(temp)
            # print(np.argmax(a_model.predict([data[110:200],to_categorical(latent_actions[110:200],4)]),axis = 1))
            # print(actions[110:200])
            # print(latent_actions[110:200])'''

    '''rew = []
    mu = pickle.load(open('Production_Models/' + 'mu' + '.dump', 'rb'))
    #episodes, n_actions = util.record_episode(env, num=1)
    for i in range(30):
        env = gym.make("BreakoutNoFrameskip-v4")
        env = util.MaxAndSkipEnv(env, 2)
        env.seed(0)
        #env = gym.wrappers.Monitor(env, 'Latent_2_Recording', force=True)
        env.reset()
        _, temp = random_play(env)
        #_, temp = latent_play(env, latent_model,a_model, guess = {0:2,1:2,2:3,3:0}, show = False)
        #_, temp = agent_play(env, c_model, mean = mu)
        rew.append(temp)
        print(temp)
    rew = np.array(rew)
    print("Mean Reward: " + str(np.mean(rew)))
    print("Std: " + str(np.std(rew)))'''