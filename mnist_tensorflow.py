"""
Script training a network on MNIST dataset using tensorflow
"""

import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np
import sklearn.utils

import utilities


class Model:

    def __init__(self):

        self.x_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 784))
        self.y_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 10))

        w = tf.Variable(tf.truncated_normal(shape=(784, 100), stddev=0.01))
        b = tf.Variable(tf.zeros(shape=100))

        z = tf.matmul(self.x_placeholder, w) + b
        a = tf.nn.relu(z)

        w = tf.Variable(tf.truncated_normal(shape=(100, 50), stddev=0.01))
        b = tf.Variable(tf.zeros(shape=50))

        z = tf.matmul(a, w) + b
        a = tf.nn.relu(z)

        w = tf.Variable(tf.truncated_normal(shape=(50, 10), stddev=0.01))
        b = tf.Variable(tf.zeros(shape=10))

        logits = tf.matmul(a, w) + b
        self.prediction = tf.nn.sigmoid(logits)

        element_wise_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_placeholder, logits=logits)
        self.loss = tf.reduce_mean(element_wise_loss)

        self.train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)


def main():

    np.set_printoptions(suppress=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data("/tmp/mnist.npz")

    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

    logger = utilities.get_logger("/tmp/mnist.html")

    # Log a few samples
    utilities.log_samples(logger, x_test, y_test)

    # Reshape 28x28 matrices to vectors 784 elements vectors
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)

    # ys are scalars, convert them to one-hot encoded vectors
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes=10)

    model = Model()

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        feed_dictionary = {model.x_placeholder: x_train_flat[:4], model.y_placeholder: y_train_categorical[:4]}

        output = session.run(model.loss, feed_dictionary)

        print(output.shape)


if __name__ == "__main__":

    main()