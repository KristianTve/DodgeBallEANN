import copy

import numpy as np

#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
import keras.backend as K
#from LiteModel import LiteModel

class Individual:

    def __init__(self, state_size, output_size):
        self.output_size = output_size
        self.state_size = state_size

        input_shape = (state_size, )

        input_layer = Input(shape=input_shape, name="inputz")
        x = BatchNormalization()(input_layer)
        x = Dense(25, activation="relu")(x)
        x = Dense(25, activation="relu")(x)
        x = Dense(25, activation="relu")(x)
        x = Dense(output_size, activation="softmax")(x)

        self.model = tf.keras.Model(input_layer, x, name="Actor")

        self.model.compile(optimizer='sgd', loss='binary_crossentropy')        # Cannot see any difference between these two in terms of error
        #self.model.compile(optimizer='sgd', loss='mse')

        print(self.model.summary())
        #self.lite_model = LiteModel.from_keras_model(self.model)

        if tf.test.gpu_device_name():
            if tf.test.is_built_with_cuda():
                print("USING GPU :)")
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

    def predict(self, x):
        return self.model.predict(x)

    def save_params(self, epochs):
        self.model.save(str(epochs))

    def load_params(self, epochs):
        #self.model = tf.keras.models.load_model("7x7/"+str(epochs))
        self.model = tf.keras.models.load_model(str(epochs))
        #self.lite_model = LiteModel.from_keras_model(self.model)    # Create a new litemodel for faster predictions

    def softmax(self, x):
        z = np.array(x) - max(x)      # Prevents under or overflow
        return (np.exp(z) / sum(np.exp(z))).tolist()

    def one_hot(self, x):
        """
        Generating a one hot encoding for the highest number in the distribution
        Uses a small random coefficient in order to select an unique one hot
        """
        rand = np.random.uniform(0.000001, 0.000009, len(x))
        x = rand+x
        max_val = np.amax(x)

        return np.where(x == max_val, 1, 0).tolist()

