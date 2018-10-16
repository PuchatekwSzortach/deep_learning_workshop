"""
Script training a network on MNIST dataset using a numpy-based neural network class
"""

import os

import cv2
import keras
import numpy as np
import sklearn.utils
import vlogging

import net
import utilities


def log_predictions(logger, model, x_data, y_data, header):

    # Display a few samples
    for image, label in zip(x_data, y_data):

        large_image = cv2.resize(image, (64, 64))

        prediction = model.predict(image.reshape(-1, 1))
        predicted_label = np.argmax(prediction)

        message = "True label: {}, predicted label: {}\nRaw predictions:\n{}".format(
            label, predicted_label, prediction)

        logger.info(vlogging.VisualRecord(header, large_image, message, fmt='jpg'))


def main():

    data_dir = "../data/"
    os.makedirs(data_dir, exist_ok=True)

    np.set_printoptions(suppress=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

    logger = utilities.get_logger(os.path.join(data_dir, "mnist.html"))

    log_size = 10

    # Log a few samples
    utilities.log_samples(logger, x_test[:log_size], y_test[:log_size])

    # Reshape 28x28 matrices to vectors 784 elements vectors
    x_train_flat = x_train.reshape(-1, 784, 1)
    x_test_flat = x_test.reshape(-1, 784, 1)

    # ys are scalars, convert them to one-hot encoded vectors
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10).reshape(-1, 10, 1)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes=10).reshape(-1, 10, 1)

    model = net.Network(layers=[784, 100, 50, 10])

    # Log untrained model predictions
    log_predictions(logger, model, x_test[:log_size], y_test[:log_size], header="Untrained model")

    model.fit(
        x_train_flat, y_train_categorical, epochs=10, learning_rate=0.1,
        x_test=x_test_flat, y_test=y_test_categorical)

    # Log trained model predictions
    log_predictions(logger, model, x_test[:log_size], y_test[:log_size], header="Trained model")


if __name__ == "__main__":

    main()
