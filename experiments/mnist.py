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

    logger = get_logger("/tmp/mnist.html")

    # Display a few samples
    for index in tqdm.tqdm(range(10)):

        image = cv2.resize(x_test[index], (64, 64))
        label = y_test[index]

        logger.info(vlogging.VisualRecord("Sample", image, str(label), fmt='jpg'))


if __name__ == "__main__":

    main()