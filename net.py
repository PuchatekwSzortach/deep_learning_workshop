"""
Module with definition of a basic neural network
"""

import numpy as np
import tqdm


def sigmoid(z):

    # Cap z to avoid overflow warnings when extremely small z leads to np.exp(-z) being infinity
    capped_z = np.maximum(z, -50)

    return 1 / (1 + np.exp(-capped_z))


def get_cost(y, a):

    return np.mean(0.5 * (y.flatten() - a.flatten())**2)


def get_statistics(model, x_data, y_data):

    costs = []

    labels = []
    predicted_labels = []

    for x, y in zip(x_data, y_data):

        prediction = model.predict(x)

        sample_cost = get_cost(y, prediction)
        costs.append(sample_cost)

        label = np.argmax(y)
        labels.append(label)

        predicted_label = np.argmax(prediction)
        predicted_labels.append(predicted_label)

    cost = np.mean(costs)
    accuracy = np.mean((np.array(labels) == np.array(predicted_labels)).astype(np.float32))

    return cost, accuracy


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
            bias_kernel = np.zeros(shape=(output_size, 1), dtype=np.float32)

            weights.append(weights_kernel)
            biases.append(bias_kernel)

        return weights, biases

    def _feed_forward(self, x):

        activations = [x]

        for index in range(self.layers_count):

            weights = self.weights[index]
            bias = self.biases[index]

            z = np.dot(weights, activations[-1]) + bias
            a = sigmoid(z)

            activations.append(a)

        return activations

    def predict(self, x):

        activations = self._feed_forward(x)
        return activations[-1]

    def _backpropagation(self, activations, y, learning_rate):

        # Make sure we work on column vectors
        activation_error = activations[-1] - y.reshape(-1, 1)

        # Lists to store parameters errors
        weights_errors = [0] * self.layers_count
        biases_errors = [0] * self.layers_count

        for index in reversed(range(self.layers_count)):

            # Use offset of +1 for activations index, since they are enumerated from input layer up
            preactivation_error = activation_error * activations[index + 1] * (1 - activations[index + 1])

            weights_errors[index] = np.dot(preactivation_error, activations[index].T)
            biases_errors[index] = preactivation_error

            activation_error = np.dot(self.weights[index].T, preactivation_error)

        for index in range(self.layers_count):

            self.weights[index] -= learning_rate * weights_errors[index]
            self.biases[index] -= learning_rate * biases_errors[index]

    def train(self, x_train, y_train, epochs, learning_rate, x_test, y_test):

        train_cost, train_accuracy = get_statistics(self, x_train, y_train)
        print("Initial training cost: {:.3f}, training accuracy: {:.3f}".format(train_cost, train_accuracy))

        test_cost, test_accuracy = get_statistics(self, x_test, y_test)
        print("Initial test cost: {:.3f}, test accuracy: {:.3f}".format(test_cost, test_accuracy))

        for epoch_index in range(epochs):

            for x, y in tqdm.tqdm(list(zip(x_train, y_train))):

                activations = self._feed_forward(x)
                self._backpropagation(activations, y, learning_rate)

            train_cost, train_accuracy = get_statistics(self, x_train, y_train)
            print("Epoch {}: training cost: {:.3f}, training accuracy: {:.3f}".format(
                epoch_index, train_cost, train_accuracy))

            test_cost, test_accuracy = get_statistics(self, x_test, y_test)
            print("Epoch {}: test cost: {:.3f}, test accuracy: {:.3f}".format(
                epoch_index, test_cost, test_accuracy))
