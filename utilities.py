"""
Utilities
"""

import os
import logging

import cv2
import vlogging


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