import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create(input_shape, sequence_length):
    inputs = tf.keras.Input(shape=(sequence_length, *input_shape))

    x = inputs
    x = layers.TimeDistributed(layers.Conv2D(64, 5, 2, activation='relu', padding='same'))(x)
    x = layers.ConvLSTM2D(128, 5, 2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(16, 3, 2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(1, 3, 2, activation='relu', padding='same')(x)
    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mean_squared_error'
    )
    return model
