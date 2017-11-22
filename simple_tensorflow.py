"""
A script with a simple tensorflow examples
"""

import numpy as np
import tensorflow as tf


def placeholders_example():

    a = tf.placeholder(dtype=tf.float32, shape=3)
    b = tf.placeholder(dtype=tf.float32, shape=3)

    c = a + b

    with tf.Session() as session:

        feed_dictionary = {
            a: np.array([2, 4, 6]),
            b: np.array([4, 8, 12])
        }

        result = session.run(c, feed_dictionary)
        print(result)

    d = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    e = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    f = d + e

    with tf.Session() as session:

        feed_dictionary = {
            d: np.array([[2, 4, 6], [2, 4, 6]]),
            e: np.array([[4, 8, 12], [-4, -8, -16]])
        }

        result = session.run(f, feed_dictionary)
        print(result)


def variables_example():

    a = tf.Variable(initial_value=[2, 4])
    b = 2 * a

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        result = session.run(b)
        print(result)


def main():

    placeholders_example()
    # variables_example()


if __name__ == "__main__":

    main()
