import numpy as np
from hebbian import Hebbian


def test_get_weights_single_image():
    k = np.array([1, -1, 1])  # [A, B, C]
    h = Hebbian(k, k)
    weights = h._get_total_weight()

    expected = 1 / 3 * np.array([
        [1, -1, 1],  # [AA, AB, AC]
        [-1, 1, -1],  # [BA, BB, BC]
        [1, -1, 1],  # [CA, CB, CC] 
    ])
    assert (weights == expected).all()


def test_get_weights_multiple_image():
    k = np.array([
        [1, 1, 1, -1, -1],  # [A, B, C, D, E]
        [1, -1, 1, -1, 1],  # [F, G, H, I, J]
    ])
    h = Hebbian(k, k)
    weights = h._get_total_weight()

    expected = 1 / 5 * np.array([
        [2, 0, 2, -2, 0],  # [AA+FF, AB+FG, AC+FH, AD+FI, AE+FJ]
        [0, 2, 0, 0, -2],  # [BA+GF, BB+GG, BC+GH, BD+GI, BE+GJ]
        [2, 0, 2, -2, 0],  # [CA+HF, CB+HG, CC+HH, CD+HI, CE+HJ]
        [-2, 0, -2, 2, 0],  # [DA+IF, DB+IG, DC+IH, DD+II, DE+IJ]
        [0, -2, 0, 0, 2],  # [EA+JF, EB+JG, EC+JH, ED+JI, EE+JJ]
    ])
    assert (weights == expected).all()


def test_get_net_input():
    k = np.array([1, -1, 1])  # [A, B, C]
    h = Hebbian(k, k)
    weights = h._get_total_weight()
    net_input = h._get_net_input(weights, k)
    assert (net_input == np.array([2 / 3, -2 / 3, 2 / 3])).all()


def test_get_activation_value_without_bias():
    k = np.array([1, -1, 1])  # [A, B, C]
    h = Hebbian(k, k)
    weights = h._get_total_weight()
    net_input = h._get_net_input(weights, k)
    activation_value = h._get_activation_value(net_input)
    assert (activation_value == np.array([2 / 3, -2 / 3, 2 / 3])).all()


def test_get_activation_value_with_bias():
    k = np.array([1, -1, 1])  # [A, B, C]
    bias = np.array([-1 / 3, 1 / 3, -1 / 3])
    h = Hebbian(k, k, bias=bias)
    weights = h._get_total_weight()
    net_input = h._get_net_input(weights, k)
    activation_value = h._get_activation_value(net_input)
    assert (activation_value == np.array([1, -1, 1])).all()


def test_sign():
    k = np.array([1, -1, 1])  # [A, B, C]
    h = Hebbian(k, k)
    weights = h._get_total_weight()
    net_input = h._get_net_input(weights, k)
    neuron_output = h._sign(net_input)
    assert (neuron_output == np.array([1, -1, 1])).all()


def test_run_converges_after_two_iterations():
    k1 = np.array([1, -1, 1])
    k2 = np.array([1, -1, -1])
    h = Hebbian(k1, k2)
    h.run()
    expected_history = np.array([
        [1, -1, -1],  # k2, initial
        [1, 1, 1],  # intermediate outcome
        [1, -1, 1],  # original image converged
    ])
    assert (h.history == expected_history).all()
    assert h.max_iterations == 8
    assert h.has_converged()
    assert not h._row_already_exists()


def test_run_does_not_converge_after_two_iterations():
    k1 = np.array([1, -1, 1])
    k2 = np.array([1, 1, -1])
    h = Hebbian(k1, k2)
    h.run()
    expected_history = np.array([
        [1, 1, -1],  # k2, initial
        [-1, 1, 1],  # intermediate outcome
        [1, 1, -1],  # k2, again
    ])
    assert (h.history == expected_history).all()
    assert h.max_iterations == 8
    assert not h.has_converged()
    assert h._row_already_exists()
