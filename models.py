# Keras
from keras import Sequential, Model, Input
from keras.utils import to_categorical
from keras.layers import Dense, Flatten,multiply, Dropout, Reshape, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras import metrics
from keras.models import model_from_json


def clipped_mse(y_true, y_pred):
        return K.mean(K.maximum(K.square(y_pred - y_true), 10), axis=-1)


def forward_model(learning_rate=0.001, decay=0.0):
        image = Input(shape=(105, 80, 6), name='image')
        action = Input(shape=(1,), name='action')
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
        model.compile(loss=clipped_mse, optimizer=Adam(lr=learning_rate, decay=decay))

        return model


def inverse_model(learning_rate=0.001, decay=0.0):
        image = Input(shape=(105, 80, 12), name='image')
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
        action = Dense(4, activation='softmax')(x)
        model = Model(inputs=[image], outputs=[action])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model


def clone_model(learning_rate=0.001, decay=0.0):
        image = Input(shape=(105, 80, 12), name='image')
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
        action = Dense(4, activation='softmax')(x)
        model = Model(inputs=[image], outputs=[action])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate, decay=decay),
                      metrics=['accuracy'])
        return model