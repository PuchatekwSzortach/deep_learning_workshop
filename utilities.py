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
    for image, label in zip(x_data, y_data):

        image = cv2.resize(image, (64, 64))
        logger.info(vlogging.VisualRecord("Sample", image, str(label), fmt='jpg'))


def get_batches_generator(x_data, y_data, batch_size):

    while True:

        shuffled_x, shuffled_y = sklearn.utils.shuffle(x_data, y_data)

        index = 0

        while index + batch_size < x_data.shape[0]:

            x_batch = shuffled_x[index: index + batch_size]
            y_batch = shuffled_y[index: index + batch_size]

            yield x_batch, y_batch

            index += batch_size
