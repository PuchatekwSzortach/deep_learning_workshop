"""
Script training a network on MNIST dataset
"""

import os
import logging

import keras
import vlogging
import tqdm
import sklearn.utils
import cv2

import network.net


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


def main():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data("/tmp/mnist.npz")

    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

    # logger = get_logger("/tmp/mnist.html")
    #
    # # Display a few samples
    # for index in tqdm.tqdm(range(10)):
    #
    #     image = cv2.resize(x_test[index], (64, 64))
    #     label = y_test[index]
    #
    #     logger.info(vlogging.VisualRecord("Sample", image, str(label), fmt='jpg'))

    # Reshape 28x28 matrices to vectors 784 elements vectors
    x_train_flat = x_train.reshape(-1, 784, 1)
    x_test_flat = x_test.reshape(-1, 784, 1)

    # ys are scalars, convert them to one-hot encoded vectors
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10)

    model = network.net.Network(layers=[784, 100, 50, 10])

    print("Input shape: {}".format(x_train_flat[0].shape))
    result = model.predict(x_train_flat[0])
    print("Output shape: {}".format(result.shape))
    print(result)


if __name__ == "__main__":

    main()