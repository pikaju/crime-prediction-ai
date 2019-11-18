import tensorflow as tf


def create(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, 5, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(16, 5, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(8, 5, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(1, 5, activation='relu', padding='same'),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
