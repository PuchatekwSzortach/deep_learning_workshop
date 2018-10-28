"""
Tests module
"""

import mock
import numpy as np

import net


def test_sigmoid():

    z = np.array([-10, -0.5, 0, 0.5, 10])

    expected = np.array([4.5397e-05, 0.3775, 0.5, 0.6224, 0.99995])
    actual = net.sigmoid(z)

    assert np.allclose(expected, actual, rtol=0.01)


def test_cost():

    y = np.array([1, 0])
    a = np.array([0.8, 0.4])

    expected = 0.05
    actual = net.get_cost(y, a)

    assert expected == actual


def test_get_statistics():

    mock_model = mock.Mock()
    mock_model.predict = mock.Mock(return_value=np.array([0.8, 0.6]).reshape(2, 1))

    # x is not important, since we use mock model with constant output
    x = np.array([0, 0]).reshape(2, 1)

    # two one-hot encoded labels
    y = np.array([[1, 0], [0, 1]])

    expected_loss = 0.15
    expected_accuracy = 0.5

    actual_loss, actual_accuracy = net.get_statistics(mock_model, x, y)

    assert np.isclose(expected_loss, actual_loss)
    assert expected_accuracy == actual_accuracy
