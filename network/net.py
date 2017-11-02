"""
Module with definition of a basic neural network
"""

import numpy as np


def sigmoid(z):

    return 1 / (1 + np.exp(-z))


def get_cost(y, a):

    return np.mean(0.5 * (y.flatten() - a.flatten())**2)


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

    def _feed_forward(self, x):

        activations = [x]
        preactivations = []

        for index in range(self.layers_count):

            weights = self.weights[index]
            bias = self.biases[index]

            z = np.dot(weights, activations[-1]) + bias
            a = sigmoid(z)

            preactivations.append(z)
            activations.append(a)

        return preactivations, activations

    def predict(self, x):

        preactivations, activations = self._feed_forward(x)
        return activations[-1]

    def _backpropagation(self, activations, preactivations, y, learining_rate):

        cost_derivative = 1

    def train(self, x_data, y_data, epochs, learning_rate):

        for x, y in zip(x_data, y_data):

            activations, preactivations = self._feed_forward(x)
            self._backpropagation(activations, preactivations, y, learning_rate)


