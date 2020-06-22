# Environment
import gym
import pybullet_envs
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
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

# NTK
import jax.numpy as npj
import jax.scipy as sp

from jax import random
from jax.experimental import optimizers
from jax.api import jit, grad, vmap

import functools

import neural_tangents as nt
from neural_tangents import stax


def train_inverse(i_model, model_path, data_path, bco_episodes = [], bco_size = 30, epoch=30, data_size=30, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False, indices = []):
    # Load all data
    if indices == []:
        indices = np.random.choice(30, data_size, replace=False)
    episodes = util.load_episodes(data_path, indices)
    data, actions = util.inverse_vector_data(episodes, frame_num=frame_num, action_num=action_num,
                                             use_images=use_images)
    # Random sample up to data_size
    data = data
    actions = actions
    # Concat
    if len(bco_episodes) != 0:
        # Load rollout data
        r_data, r_actions = util.inverse_vector_data(bco_episodes[:bco_size], frame_num=frame_num,
                                                     action_num=action_num,
                                                     use_images=use_images)
        if len(data) != 0:
            data = np.concatenate([data, r_data])
            actions = np.concatenate([actions, r_actions])
        else:
            data = r_data
            actions = r_actions
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

    return i_model, indices


def train_inverse_continuous(i_model, model_path, data_path, bco_episodes = [], bco_size = 30, epoch=30, data_size=30, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False, indices = []):
    # Load all data
    if indices == []:
        indices = np.random.choice(30, data_size, replace=False)
    episodes = util.load_episodes(data_path, indices)
    data, actions = util.inverse_vector_data(episodes, frame_num=frame_num, action_num=action_num,
                                             use_images=use_images)
    # Random sample up to data_size
    data = data.reshape((data.shape[0], -1))
    actions = actions.reshape((actions.shape[0], -1))
    # Concat
    if len(bco_episodes) != 0:
        # Load rollout data
        r_data, r_actions = util.inverse_vector_data(bco_episodes[:bco_size], frame_num=frame_num,
                                                     action_num=action_num,
                                                     use_images=use_images)
        r_data = r_data.reshape((r_data.shape[0], -1))
        r_actions = r_actions.reshape((r_actions.shape[0], -1))
        if len(data) != 0:
            data = np.concatenate([data, r_data])
            actions = np.concatenate([actions, r_actions])
        else:
            data = r_data
            actions = r_actions
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(list(actions)), list(actions))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=False, patience=500)
    mc = ModelCheckpoint('i_model.h5', monitor='val_loss', mode='min', verbose=True, save_best_only=True)
    i_model.fit([data], [actions], validation_split=0.3, batch_size=64, epochs=epoch, shuffle=True, verbose = verbose, callbacks=[es, mc])

    # serialize model to JSON
    model_json = i_model.to_json()
    with open(model_path + "i_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #i_model.save_weights(model_path + "i_model.h5")

    i_model = load_model('i_model.h5')

    return i_model, indices


def _add_diagonal_regularizer(covariance, diag_reg=0.):
  dimension = covariance.shape[0]
  reg = npj.trace(covariance) / dimension
  return covariance + diag_reg * reg * np.eye(dimension)

def _inv_operator(g_dd, diag_reg=0.0):
  g_dd_plus_reg = _add_diagonal_regularizer(g_dd, diag_reg)
  return lambda vec: sp.linalg.solve(g_dd_plus_reg, vec, sym_pos=True)

def _make_flatten_uflatten(g_td, y_train):
  """Create the flatten and unflatten utilities."""
  output_dimension = y_train.shape[-1]

  def fl(fx):
    """Flatten outputs."""
    return npj.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return npj.reshape(fx, (-1, output_dimension))

  if y_train.size > g_td.shape[-1]:
    out_dim, ragged = divmod(y_train.size, g_td.shape[-1])
    if ragged or out_dim != output_dimension:
      raise ValueError('The batch size of `y_train` must be the same as the'
                       ' last dimension of `g_td`')
    fl = lambda x: x
    ufl = lambda x: x
  return fl, ufl


def train_ntk_inverse(model_path, data_path, bco_episodes = [], bco_size = 30, epoch = 30, data_size=30, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False, indices = []):

    # Load all data
    if indices == []:
        indices = np.random.choice(30, data_size, replace=False)
    episodes = util.load_episodes(data_path, indices)
    data, actions = util.inverse_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
    #data = data[::5]
    #actions = actions[::5]

    # Concat
    if len(bco_episodes) != 0:
        # Load rollout data
        r_data, r_actions = util.inverse_vector_data(bco_episodes[:bco_size], frame_num=frame_num, action_num=action_num,
                                                     use_images=use_images)
        if len(data) != 0:
            data = np.concatenate([data,r_data])
            actions = np.concatenate([actions,r_actions])
        else:
            data = r_data
            actions = r_actions

    print(len(actions))

    '''classes = []
    n_actions = []
    n_data = []
    for i in range(action_num):
        classes.append(np.argwhere(actions == i))
        indices = np.random.choice(len(classes[i]),int(len(actions)/action_num))
        n_actions.append(actions[classes[i][indices]])
        n_data.append(data[classes[i][indices]])
    actions = np.concatenate(n_actions).reshape(-1,1)
    data = np.concatenate(n_data).reshape(-1,4)'''

    X = data
    y = to_categorical(actions, action_num)

    #from sklearn.utils import class_weight
    #class_weight = class_weight.compute_class_weight('balanced', np.unique(actions), actions)

    train_xs, test_xs, train_ys, test_ys = train_test_split(X,y, test_size = 0.0)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
        stax.Dense(action_num, W_std=1.5, b_std=0.05)
    )

    '''ntk_mean, ntk_covariance = nt.predict.gp_inference(
        kernel_fn, train_xs, train_ys, test_xs,
        diag_reg=1e-4, get='ntk', compute_cov=True)

    ntk_mean = np.reshape(ntk_mean, (-1,action_num))
    ntk_std = np.sqrt(np.diag(ntk_covariance))

    y_pred = np.argmax(ntk_mean, axis=1)
    y_true = np.argmax(test_ys, axis = 1)

    print(accuracy_score(y_true,y_pred))
    print(ntk_mean)'''

    '''kdd = kernel_fn(train_xs,train_xs,'ntk')
    ktd = kernel_fn(test_xs,train_xs,'ntk')
    op = _inv_operator(kdd, diag_reg=1e-3)
    fl, ufl = _make_flatten_uflatten(ktd, train_ys)

    mean_pred = op(fl(train_ys))
    gp = lambda test_xs: ufl(npj.dot(kernel_fn(test_xs,train_xs,'ntk'), mean_pred))

    y_pred = np.argmax(gp(test_xs), axis=1)
    y_true = np.argmax(test_ys, axis=1)

    print(accuracy_score(y_true, y_pred))'''

    gp = GaussianProcessRegressor(n_restarts_optimizer=30, alpha = 1e-2, normalize_y = True)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(train_xs, train_ys)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    episodes = util.load_episodes(data_path, range(30))
    data, actions = util.inverse_vector_data(episodes, frame_num=frame_num, action_num=action_num,
                                             use_images=use_images)
    X = data
    y = to_categorical(actions, action_num)
    y_pred, sigma = gp.predict(X, return_std=True)

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)

    print("Inverse Accuracy is: " + str(accuracy_score(y_true, y_pred)))

    return gp, indices


def train_ntk_inverse_continuous(model_path, data_path, bco_episodes = [], bco_size = 30, epoch = 30, data_size=30, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False, indices = []):

    # Load all data
    indices = np.random.choice(30, data_size, replace=False)
    episodes = util.load_episodes(data_path, indices)
    data, actions = util.inverse_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
    # Random sample up to data_size
    data = data.reshape(data.shape[0],-1)
    actions = actions.reshape(-1,action_num)
    # Concat
    if len(bco_episodes) != 0:
        # Load rollout data
        r_data, r_actions = util.inverse_vector_data(bco_episodes[:bco_size], frame_num=frame_num, action_num=action_num,
                                                     use_images=use_images)
        r_actions = r_actions.reshape(-1, action_num)
        if len(data) != 0:
            data = np.concatenate([data,r_data])
            actions = np.concatenate([actions,r_actions])
        else:
            data = r_data
            actions = r_actions

    X = data
    y = actions

    #from sklearn.utils import class_weight
    #class_weight = class_weight.compute_class_weight('balanced', np.unique(actions), actions)

    train_xs, test_xs, train_ys, test_ys = train_test_split(X,y, test_size = 0.1)

    gp = GaussianProcessRegressor(n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(train_xs, train_ys)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(test_xs, return_std=True)

    #y_pred = np.argmax(y_pred, axis=1)
    #y_true = np.argmax(test_ys, axis=1)

    #print(accuracy_score(y_true, y_pred))

    return gp, indices


def train_ntk_clone(i_model, model_path, data_path, bco_episodes = [], epoch = 30, data_size=30, pre_train_data = 1, frame_num = 4, action_num=2, use_images = False, verbose = True, indices = [], pretrain = False):

    if indices == []:
        indices = np.random.choice(30, pre_train_data, replace=False)
    dual_indices = [i for i in range(data_size) if i not in indices]

    episodes = util.load_episodes(data_path, indices)
    data, actions = util.linear_vector_data(episodes,frame_num = frame_num, action_num = action_num, use_images = use_images)
    data = data[:-frame_num+1]
    actions = actions[frame_num - 2:-1]

    #data = data[::5]
    #actions = actions[::5]

    if not pretrain:
        episodes = util.load_episodes(data_path, dual_indices)
        n_data, n_actions = util.linear_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
        n_actions = np.argmax(i_model.predict(n_data), axis = 1)
        n_data = n_data[:-frame_num + 1]
        n_actions = n_actions[frame_num - 2:-1]
        '''pred = i_model.predict(n_data)
        n_actions = []
        for i, item in enumerate(pred):
            dist = 1 / (1 - pred[i] + 0.0001) ** 2
            dist /= np.sum(dist)
            n_actions.append(np.random.choice(range(action_num), 1, p=dist)[0])'''

        data = np.concatenate([data,n_data],axis = 0)
        actions = np.concatenate([actions,n_actions])

    X = data
    y = to_categorical(actions, action_num)

    train_xs, test_xs, train_ys, test_ys = train_test_split(X,y, test_size = 0.0)

    #train_ys = to_categorical(np.argmax(i_model.predict(train_xs),axis = 1), action_num)
    #test_ys = to_categorical(np.argmax(i_model.predict(test_xs), axis=1), action_num)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
        stax.Dense(action_num, W_std=1.5, b_std=0.05)
    )

    '''ntk_mean, ntk_covariance = nt.predict.gp_inference(
        kernel_fn, train_xs, train_ys, test_xs,
        diag_reg=1e-4, get='ntk', compute_cov=True)

    ntk_mean = np.reshape(ntk_mean, (-1,action_num))
    ntk_std = np.sqrt(np.diag(ntk_covariance))

    y_pred = np.argmax(ntk_mean, axis=1)
    y_true = np.argmax(test_ys, axis = 1)

    print(accuracy_score(y_true,y_pred))
    print(ntk_mean)'''

    '''kdd = kernel_fn(train_xs,train_xs,'ntk')
    ktd = kernel_fn(test_xs,train_xs,'ntk')
    op = _inv_operator(kdd, diag_reg=1e-3)
    fl, ufl = _make_flatten_uflatten(ktd, train_ys)

    mean_pred = op(fl(train_ys))
    gp = lambda test_xs: ufl(npj.dot(kernel_fn(test_xs,train_xs,'ntk'), mean_pred))

    y_pred = np.argmax(gp(test_xs), axis=1)
    y_true = np.argmax(test_ys, axis=1)'''

    #print(accuracy_score(y_true, y_pred))

    gp = GaussianProcessRegressor(n_restarts_optimizer=30, alpha = 1e-7, normalize_y = True)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(train_xs, train_ys)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    episodes = util.load_episodes(data_path, indices)
    data, actions = util.linear_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
    X = data[:-frame_num + 1]
    y = to_categorical(actions[frame_num - 2:-1], action_num)
    y_pred, sigma = gp.predict(X, return_std=True)

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)

    print("Clone Accuracy is: " + str(accuracy_score(y_true, y_pred)))

    return gp, indices


def train_ntk_clone_continuous(i_model, model_path, data_path, bco_episodes = [], epoch = 30, data_size=30, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False):

    episodes = util.load_episodes(data_path, range(0, 0 + 30)) + bco_episodes
    data, actions = util.linear_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
    #indices = np.random.choice(len(actions), int(np.round(data_size*len(actions)/30)), replace=False)
    data = data.reshape(data.shape[0], -1)
    actions = actions.reshape(-1, action_num)
    #actions = actions[indices]
    actions = i_model.predict(data)

    X = data[:-frame_num+1]
    y = actions[frame_num - 2:-1]

    train_xs, test_xs, train_ys, test_ys = train_test_split(X,y, test_size = 0.1)

    #print(accuracy_score(y_true, y_pred))

    gp = GaussianProcessRegressor(n_restarts_optimizer=30)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(train_xs, train_ys)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(test_xs, return_std=True)

    #y_pred = np.argmax(y_pred, axis=1)
    #y_true = np.argmax(test_ys, axis=1)

    #print(accuracy_score(y_true, y_pred))

    return gp


def train_clone(c_model, i_model, model_path, data_path, bco_episodes = [], epoch = 30, data_size=30, pre_train_data = 1, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False, indices = []):

    if indices == []:
        indices = np.random.choice(30, pre_train_data, replace=False)
    dual_indices = [i for i in range(data_size) if i not in indices]

    episodes = util.load_episodes(data_path, indices)
    data, actions = util.linear_vector_data(episodes,frame_num = frame_num, action_num = action_num, use_images = use_images)
    data = data[:-frame_num+1]
    actions = actions[frame_num - 2:-1]

    #data = data[::5]
    #actions = actions[::5]

    if not pretrain:
        episodes = util.load_episodes(data_path, dual_indices)
        n_data, n_actions = util.linear_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
        n_actions = np.argmax(i_model.predict(n_data), axis = 1)
        n_data = n_data[:-frame_num + 1]
        n_actions = n_actions[frame_num - 2:-1]
        data = np.concatenate([data,n_data],axis = 0)
        actions = np.concatenate([actions,n_actions])

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

    return c_model, indices


def train_clone_continuous(c_model, i_model, model_path, data_path, bco_episodes = [], epoch = 30, data_size=30, pre_train_data=1, frame_num = 4, action_num=2, use_images = False, verbose = True, pretrain = False, indices = []):

    if indices == []:
        indices = np.random.choice(30, pre_train_data, replace=False)
    dual_indices = [i for i in range(data_size) if i not in indices]

    episodes = util.load_episodes(data_path, indices)
    data, actions = util.linear_vector_data(episodes,frame_num = frame_num, action_num = action_num, use_images = use_images)
    data = data[:-frame_num+1]
    actions = actions[frame_num - 2:-1]

    #data = data[::5]
    #actions = actions[::5]
    data = data.reshape((data.shape[0], -1))
    actions = actions.reshape((actions.shape[0], -1))

    if not pretrain:
        episodes = util.load_episodes(data_path, dual_indices)
        n_data, n_actions = util.linear_vector_data(episodes, frame_num=frame_num, action_num=action_num, use_images=use_images)
        n_data = n_data.reshape((n_data.shape[0], -1))
        n_actions = n_actions.reshape((n_actions.shape[0], -1))
        n_actions = i_model.predict(n_data)
        n_data = n_data[:-frame_num + 1]
        n_actions = n_actions[frame_num - 2:-1]
        data = np.concatenate([data,n_data],axis = 0)
        actions = np.concatenate([actions,n_actions])
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(list(actions)), list(actions))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=False, patience=500)
    mc = ModelCheckpoint('c_model.h5', monitor='val_loss', mode='min', verbose=True, save_best_only=True)
    c_model.fit([data[:-frame_num+1]], actions[frame_num-2:-1], validation_split=0.3, batch_size=64, epochs=epoch, shuffle=True, verbose = verbose, callbacks=[es, mc])

    # serialize model to JSON
    model_json = c_model.to_json()
    with open(model_path + "c_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #c_model.save_weights(model_path + "c_model.h5")

    c_model = load_model('c_model.h5')

    return c_model, indices


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
def ntk_model_evaluate(model, env, num_episodes = 30, frame_num = 4, action_num = 2, noise = 0.05, random = False):
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
            #print(np.concatenate(np.array(frames[count:count+frame_num])).reshape(1,-1))
            #dist = model(np.concatenate(np.array(frames[count:count+frame_num])).reshape(1,-1))[0]
            dist = model.predict(np.concatenate(np.array(frames[count:count+frame_num])).reshape(1,-1))[0]
            #dist -= np.min(dist)
            dist = 1 / (1 - dist + 0.0001) ** 2
            dist /= np.sum(dist)
            c_action = np.random.choice(range(action_num), 1, p=dist)[0]
            if np.random.uniform(0, 1) < noise:
                action = env.action_space.sample()
                while action == c_action:
                    action = env.action_space.sample()
            else:
                action = c_action
            #action = int(np.argmax(dist))
            if random:
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            #obs = wrap_image(env.render(mode='rgb_array'))
            #img = env.render(mode='rgb_array')
            #plt.imshow(img)
            frames.append(obs)
            episode_rewards.append(reward)
            episode[-1].insert(1,action)
            episode.append([obs, reward, done, info])
            count += 1

        env.close()
        #print(int(sum(episode_rewards)))
        #print(sum(episode_rewards))
        all_episode_rewards.append(int(sum(episode_rewards)))
        episodes.append(episode)

    #print(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    print("Reward Standard Deviation:", np.sqrt(np.var(all_episode_rewards)), "Num episodes:", num_episodes)

    return mean_episode_reward, episodes


# Push-Cart Evaluation
def ntk_model_evaluate_continuous(model, env, num_episodes = 30, frame_num = 4, action_num = 2, random = False):
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
        #print(i)
        episode_rewards = []
        episode = []
        done = False
        #obs = env.reset()
        #episode.append([obs, 0, done, None])
        count = 0
        frames = []
        #for f in range(frame_num):
        #    frames.append(obs)
        while not done:
            #print(np.concatenate(np.array(frames[count:count+frame_num])).reshape(1,-1))
            #dist = model(np.concatenate(np.array(frames[count:count+frame_num])).reshape(1,-1))[0]
            dist = model.predict(np.concatenate(np.array(frames[count:count+frame_num])).reshape(1,-1))[0]
            #dist -= np.min(dist)
            #dist = 1 / (1 - dist + 0.01) ** 2
            #dist /= np.sum(dist)
            action = dist
            #action = int(np.argmax(dist))
            if random:
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            #obs = wrap_image(env.render(mode='rgb_array'))
            #img = env.render(mode='rgb_array')
            #plt.imshow(img)
            frames.append(obs)
            episode_rewards.append(reward)
            episode[-1].insert(1,action)
            episode.append([obs, reward, done, info])
            count += 1

        env.close()
        #print(int(sum(episode_rewards)))
        #print(sum(episode_rewards))
        all_episode_rewards.append(int(sum(episode_rewards)))
        episodes.append(episode)

    #print(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    print("Reward Standard Deviation:", np.sqrt(np.var(all_episode_rewards)), "Num episodes:", num_episodes)

    return mean_episode_reward, episodes


# Push-Cart Evaluation
def model_evaluate(model, env, num_episodes = 30, frame_num = 4, action_num = 2, noise = 0.05):
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
            c_action = np.random.choice(range(action_num), 1, p=dist)[0]

            if np.random.uniform(0, 1) < noise:
                action = env.action_space.sample()
                while action == c_action:
                    action = env.action_space.sample()
            else:
                action = c_action

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
def model_evaluate_continuous(model, env, num_episodes = 30, frame_num = 4, action_num = 2):
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
        print(i)
        env = gym.make('AntBulletEnv-v0')
        obs = env.reset()
        episode_rewards = []
        episode = []
        done = False
        episode.append([obs, 0, done, None])
        count = 0
        frames = []
        for f in range(frame_num):
            frames.append(obs)
        while not done:

            dist = model.predict([[np.concatenate(np.array(frames[count:count+frame_num]))]])[0]
            #dist /= np.sum(dist)
            action = dist

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
    env = gym.make('AntBulletEnv-v0')
    obs = env.reset()
    learning_rates = [0.001, 0.001]
    dim = 28
    frame_num = 2
    action_num = 8

    pre_train_data = 1
    unlabeled_data = 30
    post_train_data = 1
    rewards = []
    i_rewards = []
    std = []
    i_std = []
    '''for k in range(1,11,3):
        print('\nUsing ' + str(k) + ' Trajectories')
        m_rewards = []
        mi_rewards = []
        pre_train_data = k
        unlabeled_data = 30
        post_train_data = 5*(15 - k)
        for j in range(5):
            # Train
            i_model, indices = train_ntk_inverse("./Basic_Models/", "./Mountain_Vector_Rollouts/", epoch=5000,
                                                 data_size=k, frame_num=frame_num,
                                                 action_num=action_num, use_images=False, verbose=False, indices=[])
            c_model, _ = train_ntk_clone(i_model, "./Basic_Models/", "./Mountain_Vector_Rollouts/", epoch=5000,
                                      data_size=30, pre_train_data = pre_train_data, frame_num =frame_num,
                                      action_num=action_num, use_images=False, verbose=False, pretrain = False, indices = indices)
            print("Pretrain")
            i_reward, new_episodes = ntk_model_evaluate(c_model, env, num_episodes=100, frame_num=frame_num,
                                                 action_num=action_num, random=False, noise = 0.00)
            for i in range(1):
                new_episodes = new_episodes[:post_train_data]
                # Remodel
                i_model, _ = train_ntk_inverse("./Basic_Models/", "./Mountain_Vector_Rollouts/", epoch=5000,
                                            data_size=k, frame_num=frame_num,
                                            action_num=action_num, bco_episodes = new_episodes, bco_size = post_train_data, use_images=False, verbose=False, indices = indices)
                c_model, _ = train_ntk_clone(i_model, "./Basic_Models/", "./Mountain_Vector_Rollouts/", epoch=5000,
                                          data_size=30, pre_train_data = pre_train_data, frame_num=frame_num,
                                          action_num=action_num, use_images=False, verbose=False, pretrain = False, indices = indices)

                print("Refined")
                reward, nn = ntk_model_evaluate(c_model, env, num_episodes=100, frame_num=frame_num,
                                                          action_num=action_num, noise = 0.00)
                print('--------------------')
                new_episodes = [nn[0]] + new_episodes

            m_rewards.append(reward)
            mi_rewards.append(i_reward)
        rewards.append(np.mean(m_rewards))
        i_rewards.append(np.mean(mi_rewards))
        std.append(np.std(m_rewards))
        i_std.append(np.std(mi_rewards))

    print(i_rewards)
    print(i_std)
    print('\n')
    print(rewards)
    print(std)
    plt.plot(i_rewards)
    plt.plot(rewards)
    plt.show()'''

    ''''# Semi-Supervised BC Linear
    for k in range(1,11,3):
        print("Using " + str(k) + " labeled trajectories and " + str(15 - k) + " rollouts.")
        episodes = []
        pre_train_data = k
        unlabeled_data = 30
        post_train_data = 15-k
        m_rewards = []
        mi_rewards = []
        for i in range(5):
            i_model = models.linear_inverse_model(learning_rate=learning_rates[0], dim=dim, frame_num=frame_num,
                                                  action_num=action_num)
            c_model = models.linear_clone_model(learning_rate=learning_rates[1], dim=dim, frame_num=frame_num,
                                                action_num=action_num)

            # Use a small collection of samples to train inverse model
            i_model, indices = train_inverse(i_model, "./Basic_Models/", "./Mountain_Vector_Rollouts/", data_size=pre_train_data,
                                    epoch=5000, frame_num=frame_num, action_num=action_num, use_images=False, verbose=False)
            # Use inverse model to label data and then train policy
            c_model, _ = train_clone(c_model, i_model, "./Basic_Models/", "./Mountain_Vector_Rollouts/", epoch=5000, data_size = unlabeled_data, pre_train_data = pre_train_data, frame_num = frame_num,
                                  action_num=action_num, use_images=False, verbose=False, pretrain = False, indices = indices)

            i_reward, n_episodes = model_evaluate(c_model, env, num_episodes=500, frame_num=frame_num,
                                                  action_num=action_num)

            episodes += n_episodes[:post_train_data]
            # Use a small collection of samples to train inverse model
            i_model, _ = train_inverse(i_model, "./Basic_Models/", "./Mountain_Vector_Rollouts/", data_size=pre_train_data,
                                    epoch=5000, frame_num=frame_num, bco_episodes=episodes, bco_size = post_train_data, action_num = action_num, use_images=False, verbose=False, indices = indices)
            # Use inverse model to label data and then train policy
            c_model, _ = train_clone(c_model, i_model, "./Basic_Models/", "./Mountain_Vector_Rollouts/", epoch=5000,
                                  data_size=unlabeled_data, pre_train_data = pre_train_data, frame_num=frame_num,
                                  action_num=action_num, use_images=False, verbose=False, indices = indices, pretrain = False)

            reward, new_episodes = model_evaluate(c_model, env, num_episodes=500, frame_num=frame_num,
                                                  action_num=action_num)
            mi_rewards.append(i_reward)
            m_rewards.append(reward)
        i_rewards.append(np.mean(mi_rewards))
        rewards.append(np.mean(m_rewards))
        i_std.append(np.std(mi_rewards))
        std.append(np.std(m_rewards))

    print(i_rewards)
    print(i_std)
    print('\n')
    print(rewards)
    print(std)
    plt.plot(i_rewards)
    plt.plot(rewards)
    plt.show()'''

    # Semi-Supervised BC Linear Continuous
    for k in range(5, 11, 3):
        print("Using " + str(k) + " labeled trajectories and " + str(15 - k) + " rollouts.")
        episodes = []
        pre_train_data = k
        unlabeled_data = 30
        post_train_data = 15 - k
        m_rewards = []
        mi_rewards = []
        for i in range(5):
            i_model = models.linear_inverse_model_continuous(learning_rate=learning_rates[0], dim=dim, frame_num=frame_num,
                                                  action_num=action_num)
            c_model = models.linear_clone_model_continuous(learning_rate=learning_rates[1], dim=dim, frame_num=frame_num,
                                                action_num=action_num)

            # Use a small collection of samples to train inverse model
            i_model, indices = train_inverse_continuous(i_model, "./Basic_Models/", "./Ant_Vector_Rollouts/",
                                             data_size=pre_train_data,
                                             epoch=5, frame_num=frame_num, action_num=action_num, use_images=False,
                                             verbose=False)
            # Use inverse model to label data and then train policy
            c_model, _ = train_clone_continuous(c_model, i_model, "./Basic_Models/", "./Ant_Vector_Rollouts/", epoch=5,
                                     data_size=unlabeled_data, pre_train_data=pre_train_data, frame_num=frame_num,
                                     action_num=action_num, use_images=False, verbose=False, pretrain=False,
                                     indices=indices)

            i_reward, n_episodes = model_evaluate_continuous(c_model, env, num_episodes=2, frame_num=frame_num,
                                                  action_num=action_num)

            episodes += n_episodes[:post_train_data]
            # Use a small collection of samples to train inverse model
            i_model, _ = train_inverse_continuous(i_model, "./Basic_Models/", "./Ant_Vector_Rollouts/",
                                       data_size=pre_train_data,
                                       epoch=50, frame_num=frame_num, bco_episodes=episodes, bco_size=post_train_data,
                                       action_num=action_num, use_images=False, verbose=False, indices=indices)
            # Use inverse model to label data and then train policy
            c_model, _ = train_clone_continuous(c_model, i_model, "./Basic_Models/", "./Ant_Vector_Rollouts/", epoch=50,
                                     data_size=unlabeled_data, pre_train_data=pre_train_data, frame_num=frame_num,
                                     action_num=action_num, use_images=False, verbose=False, indices=indices,
                                     pretrain=False)

            reward, new_episodes = model_evaluate_continuous(c_model, env, num_episodes=2, frame_num=frame_num,
                                                  action_num=action_num)
            mi_rewards.append(i_reward)
            m_rewards.append(reward)
        i_rewards.append(np.mean(mi_rewards))
        rewards.append(np.mean(m_rewards))
        i_std.append(np.std(mi_rewards))
        std.append(np.std(m_rewards))

    print(i_rewards)
    print(i_std)
    print('\n')
    print(rewards)
    print(std)
    plt.plot(i_rewards)
    plt.plot(rewards)
    plt.show()

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
