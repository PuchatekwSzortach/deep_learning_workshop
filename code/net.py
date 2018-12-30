"""
Module with definition of a basic neural network
"""

import numpy as np
import tqdm
import sklearn.utils


def sigmoid(z):

    # Cap z to avoid overflow warnings when extremely small z leads to np.exp(-z) being infinity
    capped_z = np.maximum(z, -50)

    return 1 / (1 + np.exp(-capped_z))


def get_cost(y, a):

    return np.mean(0.5 * (y - a)**2)


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

        self.layers_count = len(layers)
        self.weights, self.biases = self._initialize_parameters(layers)

    def _initialize_parameters(self, layers):

        weights = [None] * self.layers_count
        biases = [None] * self.layers_count

        for index in range(1, self.layers_count):

            input_size = layers[index - 1]
            output_size = layers[index]

            weights_kernel = np.random.randn(output_size, input_size)
            bias_kernel = np.zeros(shape=(output_size, 1), dtype=np.float32)

            weights[index] = weights_kernel
            biases[index] = bias_kernel

        return weights, biases

    def _feed_forward(self, x):

        activations = [x]

        for index in range(1, self.layers_count):

            z = np.dot(self.weights[index], activations[index - 1]) + self.biases[index]
            a = sigmoid(z)

            activations.append(a)

        return activations

    def predict(self, x):

        activations = self._feed_forward(x)
        return activations[-1]

    def _backpropagation(self, activations, y, learning_rate):

        activation_errors = [None] * self.layers_count
        preactivation_errors = [None] * self.layers_count

        # Lists to store parameters errors
        weights_errors = [None] * self.layers_count
        biases_errors = [None] * self.layers_count

        # Get cost derivative w.r.t. to output layer activations
        activation_errors[-1] = (activations[-1] - y) / y.size

        for index in reversed(range(1, self.layers_count)):

            preactivation_errors[index] = activation_errors[index] * activations[index] * (1 - activations[index])

            weights_errors[index] = np.dot(preactivation_errors[index], activations[index - 1].T)
            biases_errors[index] = preactivation_errors[index]

            activation_errors[index - 1] = np.dot(self.weights[index].T, preactivation_errors[index])

        for index in range(1, self.layers_count):

            self.weights[index] -= learning_rate * weights_errors[index]
            self.biases[index] -= learning_rate * biases_errors[index]

    def fit(self, x_train, y_train, epochs, learning_rate, x_test, y_test):

        for epoch_index in range(epochs):

            x_train, y_train = sklearn.utils.shuffle(x_train, y_train)

            for x, y in tqdm.tqdm(list(zip(x_train, y_train))):

                activations = self._feed_forward(x)
                self._backpropagation(activations, y, learning_rate)

            train_cost, train_accuracy = get_statistics(self, x_train, y_train)
            print("Epoch {}: training cost: {:.3f}, training accuracy: {:.3f}".format(
                epoch_index, train_cost, train_accuracy))

            test_cost, test_accuracy = get_statistics(self, x_test, y_test)
            print("Epoch {}: test cost: {:.3f}, test accuracy: {:.3f}".format(
                epoch_index, test_cost, test_accuracy))
