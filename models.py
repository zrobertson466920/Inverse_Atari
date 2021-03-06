# Keras
import tensorflow as tf
import keras
from keras import Sequential, Model, Input
from keras.utils import to_categorical
from keras.layers import Dense, Flatten,multiply, Dropout, Reshape, Activation, Lambda, Dot
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import metrics
from keras.models import model_from_json

import numpy as np

# Various Losses in various states of use and disuse
def clipped_mse(y_true, y_pred):
        return K.mean(K.maximum(K.square(y_pred - y_true), 10), axis=-1)


# Has not been tested
# ...well actually it has the correct shape and reduces properly
# Basically, I'm using a lambda function with tf to split my keras tensor
def temp_mse(noise):

    def min_mse(y_true, y_pred):
        dist = K.mean(K.square(y_pred - y_true),(2,3,4),keepdims = True)[:,:,0,0,0]
        dist = dist*K.random_uniform((4,),minval = 1-noise, maxval = 1+noise)
        return K.min(dist,axis = 1)

    return min_mse


def min_mse(y_true, y_pred):
    dist = K.mean(K.square(y_pred - y_true),(2,3,4),keepdims = True)[:,:,0,0,0]
    return K.min(dist,axis = 1)


def vector_min_mse(y_true, y_pred):
    dist = K.mean(K.square(y_pred - y_true), (2,), keepdims=True)[:, :, 0]
    return K.min(dist, axis=1)


def p_mse(new_img,a_img,p_image):

    def loss(y_true,y_pred):
        index = K.cast(K.argmin(K.mean(K.square(new_img - a_img), (2, 3, 4), keepdims=True)[:,:,0,0,0],axis = 1)[0],dtype = 'int64')
        return keras.losses.mse(new_img[:,index,:,:,:],p_image)

    return loss


# Only takes in numpy (conversions result in memory leak)


def argmin_mse(y_true, y_pred):
    val = np.argmin(np.mean(np.square(y_pred - y_true), (2, 3, 4), keepdims=True)[:,:,0,0,0],axis = 1)
    return val


def vector_argmin_mse(y_true, y_pred):
    val = np.argmin(np.mean(np.square(y_pred - y_true), (2,), keepdims=True)[:,:,0],axis = 1)
    return val


def w_sum(arg):
    return K.sum(arg[0] * K.tile(Reshape((4, 1, 1, 1))(arg[1]), (1, 1, 105, 80, 6)), axis=1)


def latent_cross(new_img,af_img):

    def metric(y_true,y_pred):
        val = K.cast(K.one_hot(K.argmin(K.mean(K.square(new_img-af_img), (2,3,4), keepdims=True)[:,:,0,0,0],axis = 1),4),dtype ='float32')
        return K.categorical_crossentropy(val,y_pred)

    return metric


def latent_acc(new_img,af_img):

    def metric(y_true,y_pred):
        val = K.cast(K.one_hot(K.argmin(K.mean(K.square(new_img - af_img), (2, 3, 4), keepdims=True)[:, :, 0, 0, 0], axis=1), 4),dtype='float32')
        return keras.metrics.categorical_accuracy(val,y_pred)

    return metric


# ILFO Models
def modal_model(learning_rate=0.001, decay=0.0, l_num = 4):

    # Forward Prediction
    image = Input(shape=(105, 80, 12), name='image')
    after_image = Input(shape = (l_num,105,80,6), name = 'after_image')
    x = Conv2D(64, (4, 4), strides=2, activation='relu', input_shape=(105, 80, 6))(image)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='linear')(x)
    x = Dense(512, activation='linear')(x)
    x = Dropout(0.2)(x)
    x = Dense(2560, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Reshape((5, 4, 128))(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu')(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu')(x)
    x = Conv2DTranspose(64, (6, 3), strides=2, activation='relu')(x)
    x = Conv2DTranspose(6*l_num, (7, 4), strides=2, activation='relu')(x)
    new_image = Reshape((l_num,105,80,6),name = 'new_image')(x)

    model = Model(inputs=[image], outputs=[new_image])
    model.compile(loss=[min_mse], loss_weights=[1.0], optimizer=Adam(lr=learning_rate, decay=decay))

    return model


def latent_model(learning_rate = 0.001, decay = 0.0, l_num = 4):

    # Latent Prediction
    image = Input(shape=(105, 80, 12), name='image')
    x = Conv2D(64, (4, 4), strides=2, activation='relu', input_shape=(105, 80, 6))(image)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', name = 'embedding')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Activation('sigmoid')(x)
    action = Dense(l_num, activation='softmax', name = 'action')(x)

    model = Model(inputs=[image], outputs=[action])
    model.compile(loss=['categorical_crossentropy'], loss_weights = [1.0], metrics = {'action': 'categorical_accuracy'}, optimizer=Adam(lr=learning_rate, decay=decay))

    return model


def action_model(learning_rate = 0.001, decay = 0.0, l_num = 4, a_num = 4):

    image = Input(shape=(105, 80, 12), name='image')
    l_action = Input(shape=(l_num,), name='l_action')
    x = Conv2D(64, (4, 4), strides=2, activation='relu', input_shape=(105, 80, 12))(image)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    y = Dense(256, activation='linear')(l_action)
    x = multiply([x, y])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Activation('sigmoid')(x)
    action = Dense(a_num, activation='softmax')(x)
    model = Model(inputs=[image, l_action], outputs=[action])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                  metrics=['accuracy'])
    return model


# Used for embedding layers
def alt_action_model(model, learning_rate = 0.001, decay = 0.0, l_num = 4, a_num = 4):

    image = Input(shape=(105, 80, 12), name='image')
    layer_name = 'embedding'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    l_action = Input(shape=(l_num,), name='l_action')

    x = intermediate_layer_model(image)
    y = Dense(512, activation='linear')(l_action)
    x = multiply([x, y])
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(4, activation='relu')(x)
    x = Activation('sigmoid')(x)
    action = Dense(a_num, activation='softmax')(x)
    model = Model(inputs=[image,l_action], outputs=[action])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                  metrics=['accuracy'])
    return model


# Prototype
def forward_model(learning_rate=0.001, decay=0.0):
        image = Input(shape=(105, 80, 6), name='image')
        action = Input(shape=(4,), name='action')
        x = Conv2D(64, (4, 4), strides=2, activation='relu', input_shape=(105, 80, 6))(image)
        x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
        x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
        x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='linear')(x)
        y = Dense(1024, activation='linear')(action)
        x = multiply([x, y])
        x = Dense(512, activation='linear')(x)
        x = Dropout(0.2)(x)
        x = Dense(2560, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Reshape((5, 4, 128))(x)
        x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu')(x)
        x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu')(x)
        x = Conv2DTranspose(64, (6, 3), strides=2, activation='relu')(x)
        new_image = Conv2DTranspose(6, (7, 4), strides=2, activation='relu')(x)
        model = Model(inputs=[image, action], outputs=[new_image])
        model.compile(loss=min_mse, optimizer=Adam(lr=learning_rate, decay=decay))

        return model


# BCO Models
def inverse_model(learning_rate=0.001, decay=0.0, frame_num = 4, action_num = 4):
        image = Input(shape=(105, 80, 3*frame_num), name='image')
        x = Conv2D(64, (4, 4), strides=2, activation='relu', input_shape=(105, 80, 12))(image)
        x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=2, activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=2, activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('sigmoid')(x)
        action = Dense(action_num, activation='softmax')(x)
        model = Model(inputs=[image], outputs=[action])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model


def clone_model(learning_rate=0.001, decay=0.0, frame_num=4, action_num=2):
    image = Input(shape=(105, 80, 3 * frame_num), name='image')
    x = Conv2D(64, (4, 4), strides=2, activation='relu', input_shape=(105, 80, 12))(image)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    action = Dense(action_num, activation='softmax')(x)
    model = Model(inputs=[image], outputs=[action])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                  metrics=['accuracy'])
    return model


# Vector Environment Networks

# Vector BCO
'''def linear_clone_model(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, action_num = 2):

        image = Input(shape = (frame_num * dim,), name='image')
        x = Dense(2, activation = 'relu')(image)
        action = Dense(action_num, activation = 'softmax')(x)
        model = Model(inputs=[image], outputs=[action])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model


def linear_inverse_model(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, action_num = 2):

        image = Input(shape=(frame_num * dim,), name='image')
        x = Dense(action_num, activation = 'relu')(image)
        action = Dense(action_num, activation='softmax')(x)
        model = Model(inputs=[image], outputs=[action])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model'''


# Non-Linear Vector BCO
def linear_clone_model(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, action_num = 2):

        image = Input(shape = (frame_num * dim,), name='image')
        x = Dense(50*action_num, activation = 'relu')(image)
        x = Dense(action_num, activation='relu')(x)
        action = Dense(action_num, activation='softmax')(x)
        model = Model(inputs=[image], outputs=[action])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model

# Non-Linear Vector BCO
def linear_clone_model_continuous(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, action_num = 2):

        image = Input(shape = (frame_num * dim,), name='image')
        x = Dense(32, activation = 'relu')(image)
        x = Dense(32, activation='relu')(x)
        x = Dense(action_num, activation='relu')(x)
        action = Dense(action_num, activation='softmax')(x)
        model = Model(inputs=[image], outputs=[action])

        model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model


def linear_inverse_model(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, action_num = 2):

        image = Input(shape=(frame_num * dim,), name='image')
        x = Dense(50*action_num, activation = 'relu')(image)
        x = Dense(action_num, activation='relu')(x)
        action = Dense(action_num, activation='softmax')(x)
        model = Model(inputs=[image], outputs=[action])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model


def linear_inverse_model_continuous(learning_rate=0.001, decay=0.0, dim=4, frame_num=4, action_num=2):
    image = Input(shape=(frame_num * dim,), name='image')
    x = Dense(100, activation='relu')(image)
    x = Dense(100, activation='relu')(x)
    x = Dense(action_num, activation='relu')(x)
    action = Dense(action_num, activation='softmax')(x)
    model = Model(inputs=[image], outputs=[action])

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay),
                  metrics=['accuracy'])
    return model


# Vector ILFO
def vector_modal_model(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, latent_action_num = 2):

    # Forward Prediction
    image = Input(shape=(frame_num * dim,), name='image')
    x = Dense(latent_action_num * dim, activation = 'relu')(image)
    x = Dense(latent_action_num * frame_num * dim*10, activation='relu')(x)
    x = Dense(latent_action_num * dim, activation='relu')(x)
    new_image = Reshape((latent_action_num, dim), name='new_image')(x)

    model = Model(inputs=[image], outputs=[new_image])
    model.compile(loss=[vector_min_mse], loss_weights=[1.0], optimizer=Adam(lr=learning_rate, decay=decay))

    return model


def vector_latent_model(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, latent_action_num = 2):

    # Latent Prediction
    image = Input(shape=(frame_num * dim,), name='image')
    x = Dense(latent_action_num, activation='relu')(image)
    action = Dense(latent_action_num, activation='softmax')(x)

    model = Model(inputs=[image], outputs=[action])
    model.compile(loss=['categorical_crossentropy'], loss_weights=[1.0], metrics={'action': 'categorical_accuracy'},
                  optimizer=Adam(lr=learning_rate, decay=decay))

    return model


def vector_action_model(learning_rate = 0.001, decay = 0.0, dim = 4, frame_num = 4, latent_action_num = 2, action_num = 2):

    image = Input(shape=(frame_num * dim,), name='image')
    latent_action = Input(shape=(latent_action_num,), name='l_action')
    x = Dense(latent_action_num * frame_num, activation='relu')(image)
    y = Dense(latent_action_num * frame_num, activation='linear')(latent_action)
    x = multiply([x, y])
    x = Dense(action_num, activation='relu')(x)
    action = Dense(action_num, activation='softmax')(x)
    model = Model(inputs=[image, latent_action], outputs=[action])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                  metrics=['accuracy'])
    return model
