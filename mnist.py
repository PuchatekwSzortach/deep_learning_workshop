"""
Script training a network on MNIST dataset using a numpy-based neural network class
"""

import cv2
import keras
import numpy as np
import sklearn.utils
import vlogging

import net
import utilities


def log_predictions(logger, model, x_data, y_data, header):

    # Display a few samples
    for index in range(10):

        image = cv2.resize(x_data[index], (64, 64))
        label = y_data[index]

        prediction = model.predict(x_data[index].reshape(-1, 1))
        predicted_label = np.argmax(prediction)

        message = "True label: {}, predicted label: {}\nRaw predictions:\n{}".format(
            label, predicted_label, prediction)

        logger.info(vlogging.VisualRecord(header, image, message, fmt='jpg'))


def main():

    np.set_printoptions(suppress=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data("/tmp/mnist.npz")

    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

    logger = utilities.get_logger("/tmp/mnist.html")

    # Log a few samples
    utilities.log_samples(logger, x_test, y_test)

    # Reshape 28x28 matrices to vectors 784 elements vectors
    x_train_flat = x_train.reshape(-1, 784, 1)
    x_test_flat = x_test.reshape(-1, 784, 1)

    # ys are scalars, convert them to one-hot encoded vectors
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10).reshape(-1, 10, 1)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes=10).reshape(-1, 10, 1)

    model = net.Network(layers=[784, 100, 50, 10])

    # Log untrained model predictions
    log_predictions(logger, model, x_test, y_test, header="Untrained model")

    model.train(
        x_train_flat, y_train_categorical, epochs=10, learning_rate=0.1,
        x_test=x_test_flat, y_test=y_test_categorical)

    # Log trained model predictions
    log_predictions(logger, model, x_test, y_test, header="Trained model")


if __name__ == "__main__":

    main()
