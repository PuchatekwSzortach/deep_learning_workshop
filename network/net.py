"""
Module with definition of a basic neural network
"""

import numpy as np


def sigmoid(z):

    return 1 / (1 + np.exp(z))


class Network:

    def __init__(self, layers):

        # Subtract input layer
        self.layers_count = len(layers) - 1
        self.weights, self.biases = self._initialize_parameters(layers)

    def _initialize_parameters(self, layers):

        weights = []
        biases = []

        for input_size, output_size in zip(layers[:-1], layers[1:]):

            weights_kernel = np.random.randn(output_size, input_size)
            bias_kernel = np.zeros(shape=(output_size,1), dtype=np.float32)

            weights.append(weights_kernel)
            biases.append(bias_kernel)

        return weights, biases

    def predict(self, x):

        a = x.copy()

        for index in range(self.layers_count):

            weights = self.weights[index]
            bias = self.biases[index]

            z = np.dot(weights, a) + bias
            a = sigmoid(z)

        return a


