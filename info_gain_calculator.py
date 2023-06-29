import numpy as np


def calculate_probability(y: np.ndarray):
    size = y.shape[0]
    if size == 0:
        return 0.0
    p = 0
    for training_example in y:
        if training_example == 1:
            p += 1

    return p / size


def calculate_entropy(y: np.ndarray):
    p = calculate_probability(y)

    if p == 0 or p == 1:
        entropy = 0.0
    else:
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    return entropy


def calculate_info_gain(root_entropy, y_left: np.ndarray, y_right: np.ndarray):
    left_size = y_left.shape[0]
    right_size = y_right.shape[0]

    if left_size == 0 or right_size == 0:
        return 0.0

    total_size = left_size + right_size

    h_left = calculate_entropy(y_left)
    h_right = calculate_entropy(y_right)

    w_left = left_size / total_size
    w_right = right_size / total_size

    info_gain = root_entropy - (w_left * h_left + w_right * h_right)

    return info_gain


def measure_accuracy(y_predicted: np.ndarray, y_test: np.ndarray):
    size = y_predicted.shape[0]
    true_results = 0

    if size != y_test.shape[0]:
        raise Exception("Error in shape")

    for example_num in range(size):
        if y_predicted[example_num] == y_test[example_num]:
            true_results += 1

    accuracy = true_results / size

    return accuracy
