from info_gain_calculator import *


def split_data(x: np.ndarray, y: np.ndarray, feature_id: int):
    if feature_id == -1 or feature_id >= x.shape[1]:
        return None, None, None, None

    y_left = []
    y_right = []

    left_indices = []
    right_indices = []

    x_trans = x.T
    wanted_x = x_trans[feature_id]

    for training_example_num in range(len(wanted_x)):
        if wanted_x[training_example_num]:
            y_left.append(y[training_example_num])
            left_indices.append(training_example_num)
        else:
            y_right.append(y[training_example_num])
            right_indices.append(training_example_num)

    x_left_np = x[left_indices]
    x_right_np = x[right_indices]

    y_left_np = np.array(y_left)
    y_right_np = np.array(y_right)

    return x_left_np, x_right_np, y_left_np, y_right_np


def splitting_criteria_met(max_depth, cur_depth, y, best_feature_id, info_gain):
    if cur_depth >= max_depth or best_feature_id == -1 or info_gain == 0:
        return True

    if calculate_probability(y) == 1 or calculate_probability(y) == 0:
        return True

    return False
