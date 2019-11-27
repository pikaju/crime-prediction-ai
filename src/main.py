import tensorflow as tf
import numpy as np
import heatmap
import predictor


def main():
    data = np.load('heatmaps.npy')
    data = data[..., tf.newaxis]

    sequence_length = 4

    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i])
        y.append(data[i])
    x, y = np.array(x), np.array(y)

    model = predictor.create(data[0].shape, sequence_length)

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
        stacked = np.hstack((yt, yp))
        heatmap.save('../frames/frame_{}.png'.format(i),
                     stacked.reshape(stacked.shape[:-1]))


if __name__ == '__main__':
    main()
