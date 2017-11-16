"""
Script training a network on MNIST dataset
"""

import logging
import os

import cv2
import keras
import numpy as np
import sklearn.utils
import vlogging

import net


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to log.html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("mnist")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def log_samples(logger, x_data, y_data):

    # Display a few samples
    for index in range(10):

        image = cv2.resize(x_data[index], (64, 64))
        label = y_data[index]

        logger.info(vlogging.VisualRecord("Sample", image, str(label), fmt='jpg'))


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

    logger = get_logger("/tmp/mnist.html")

    # Log a few samples
    log_samples(logger, x_test, y_test)

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
        x_train_flat, y_train_categorical, epochs=10, learning_rate=0.01,
        x_test=x_test_flat, y_test=y_test_categorical)

    # Log trained model predictions
    log_predictions(logger, model, x_test, y_test, header="Trained model")


if __name__ == "__main__":

    main()
