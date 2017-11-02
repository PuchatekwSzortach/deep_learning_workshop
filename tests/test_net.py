"""
Tests module
"""

import numpy as np

import network.net


def test_sigmoid():

    z = np.array([-10, -0.5, 0, 0.5, 10])

    expected = np.array([4.5397e-05, 0.3775, 0.5, 0.6224, 0.99995])
    actual = network.net.sigmoid(z)

    assert np.allclose(expected, actual, rtol=0.01)


def test_cost():

    y = np.array([1, 0])
    a = np.array([0.8, 0.4])

    expected = 0.05
    actual = network.net.get_cost(y, a)

    assert expected == actual
