"""
Utilities
"""

import os
import logging

import cv2
import vlogging
import sklearn.utils


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


def get_batches_generator(x, y, batch_size):

    while True:

        shuffled_x, shuffled_y = sklearn.utils.shuffle(x, y)

        index = 0

        while index + batch_size < x.shape[0]:

            x_batch = shuffled_x[index: index + batch_size]
            y_batch = shuffled_y[index: index + batch_size]

            yield x_batch, y_batch

            index += batch_size
