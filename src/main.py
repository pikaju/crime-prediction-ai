import tensorflow as tf
import numpy as np
import heatmap
import predictor


loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


def main():
    data = np.load('heatmaps.npy')
    data = data[..., tf.newaxis]

    model = predictor.create(data[0].shape)

    x, y = data[:-1], data[1:]
    split = len(x) // 5
    x_train = x[:-split]
    y_train = y[:-split]
    x_test = x[-split:]
    y_test = y[-split:]

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              validation_split=0.2)

    print('Generating output images...')

    y_pred = model.predict(x_test)
    for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
        stacked = np.vstack((yt, yp))
        heatmap.save('frames/frame_{}.png'.format(i),
                     stacked.reshape(stacked.shape[:-1]))


if __name__ == '__main__':
    main()
