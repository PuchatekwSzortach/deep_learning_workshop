"""
Script training a network on MNIST dataset using tensorflow
"""

import os

import tensorflow as tf
import tensorflow.contrib.keras as keras
import numpy as np
import sklearn.utils
import tqdm
import cv2
import vlogging

import utilities


class Model:

    def __init__(self):

        self.x_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 784))
        self.y_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 10))

        w1 = tf.Variable(tf.truncated_normal(shape=(784, 100), stddev=0.01))
        b1 = tf.Variable(tf.zeros(shape=100))

        z1 = tf.matmul(self.x_placeholder, w1) + b1
        a1 = tf.nn.relu(z1)

        w2 = tf.Variable(tf.truncated_normal(shape=(100, 50), stddev=0.01))
        b2 = tf.Variable(tf.zeros(shape=50))

        z2 = tf.matmul(a1, w2) + b2
        a2 = tf.nn.relu(z2)

        w3 = tf.Variable(tf.truncated_normal(shape=(50, 10), stddev=0.01))
        b3 = tf.Variable(tf.zeros(shape=10))

        z3 = tf.matmul(a2, w3) + b3
        a3 = tf.nn.softmax(z3)

        self.prediction = a3

        self.loss_op = tf.losses.softmax_cross_entropy(onehot_labels=self.y_placeholder, logits=z3)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss_op)

    def fit(self, session, x_train, y_train, epochs, x_test, y_test, batch_size):

        training_set_size = x_train.shape[0]
        training_batches_generator = utilities.get_batches_generator(x_train, y_train, batch_size)

        for epoch_index in range(epochs):

            print("Epoch {}".format(epoch_index))

            for _ in tqdm.tqdm(range(training_set_size // batch_size)):

                x_batch, y_batch = next(training_batches_generator)

                feed_dictionary = {self.x_placeholder: x_batch, self.y_placeholder: y_batch}
                session.run(self.train_op, feed_dictionary)

            training_loss, training_accuracy = get_statistics(
                session, self, x_train, y_train, batch_size)

            print("Epoch {}: training loss: {:.3f}, training accuracy: {:.3f}".format(
                epoch_index, training_loss, training_accuracy))

            test_loss, test_accuracy = get_statistics(
                session, self, x_test, y_test, batch_size)

            print("Epoch {}: test loss: {:.3f}, test accuracy: {:.3f}".format(epoch_index, test_loss, test_accuracy))


def log_predictions(logger, model, session, x_data, y_data, header):
    """
    Log model predictions along with correct answers
    :param logger: logger instance
    :param model: model
    :param session: tensorflow session
    :param x_data: batch of MNIST images
    :param y_data: batch of MNIST labels
    :param header: header for log entries
    """

    data_size = x_data.shape[0]

    feed_dictionary = {model.x_placeholder: x_data.reshape(data_size, 784)}
    prediction = session.run(model.prediction, feed_dictionary)

    for index in range(data_size):

        image = cv2.resize(x_data[index], (64, 64))

        label = y_data[index]
        predicted_label = np.argmax(prediction[index])

        message = "True label: {}, predicted label: {}\nRaw predictions:\n{}".format(
            label, predicted_label, prediction[index].reshape(10, 1))

        logger.info(vlogging.VisualRecord(header, image, message, fmt='jpg'))


def get_statistics(session, model, x_data, y_data, batch_size):
    """
    Compute model's loss and accuracy over data
    :param session: tensorflow session
    :param model: model
    :param x_data: mnist images in vector format
    :param y_data: one-hot encoded mnist labels
    :param batch_size:
    :return: mean loss and accuracy computed over whole dataset
    """

    losses = []

    labels = []
    predicted_labels = []

    batches_generator = utilities.get_batches_generator(x_data, y_data, batch_size)
    data_size = x_data.shape[0]

    for _ in range(data_size // batch_size):

        x_batch, y_batch = next(batches_generator)

        feed_dictionary = {
            model.x_placeholder: x_batch,
            model.y_placeholder: y_batch
        }

        batch_loss, batch_prediction = session.run([model.loss_op, model.prediction], feed_dictionary)

        losses.append(batch_loss)

        labels_batch = np.argmax(y_batch, axis=1).flatten()
        labels.extend(labels_batch)

        predicted_batch_labels = np.argmax(batch_prediction, axis=1).flatten()
        predicted_labels.extend(predicted_batch_labels)

    loss = np.mean(losses)
    accuracy = np.mean((np.array(labels) == np.array(predicted_labels)).astype(np.float32))

    return loss, accuracy


def main():

    data_dir = "../data/"
    os.makedirs(data_dir, exist_ok=True)

    np.set_printoptions(suppress=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

    logger = utilities.get_logger(os.path.join(data_dir, "mnist_tensorflow.html"))

    log_size = 10

    # Log a few samples
    utilities.log_samples(logger, x_test[:log_size], y_test[:log_size])

    # Reshape 28x28 matrices to 784 elements vectors
    x_train_flat = x_train.reshape(60000, 784)
    x_test_flat = x_test.reshape(10000, 784)

    # ys are scalars, convert them to one-hot encoded vectors
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes=10)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes=10)

    model = Model()

    batch_size = 32
    epochs = 10

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        # Log untrained model predictions
        log_predictions(logger, model, session, x_test[:log_size], y_test[:log_size], header="Untrained model")

        training_loss, training_accuracy = get_statistics(
            session, model, x_train_flat, y_train_categorical, batch_size)

        print("Initial training loss: {:.3f}, training accuracy: {:.3f}".format(training_loss, training_accuracy))

        test_loss, test_accuracy = get_statistics(
            session, model, x_test_flat, y_test_categorical, batch_size)

        print("Initial test loss: {:.3f}, test accuracy: {:.3f}".format(test_loss, test_accuracy))

        model.fit(session, x_train_flat, y_train_categorical, epochs, x_test_flat, y_test_categorical, batch_size)

        # Log trained model predictions
        log_predictions(logger, model, session, x_test[:log_size], y_test[:log_size], header="Trained model")


if __name__ == "__main__":

    main()
