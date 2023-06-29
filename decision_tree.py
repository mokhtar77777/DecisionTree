from data_splitter import *
from tree_formator import *


class DecisionNode:
    def __init__(self):
        self.next_nodes = [None, None]
        self.feature_id = -1


class LeafNode:
    def __init__(self):
        self.prediction = None


class DecisionTreeClassifier:
    def __init__(self):
        self.root_node = None
        self.input_shape = -1

    def _create_leaf_node(self, parent: DecisionNode, cur_class):
        leaf_node = LeafNode()
        leaf_node.prediction = cur_class
        if parent is None:
            self.root_node = leaf_node
        else:
            parent.next_nodes[cur_class] = leaf_node

    def _create_decision_node(self, parent: DecisionNode, cur_class, feature_id):
        decision_node = DecisionNode()
        decision_node.feature_id = feature_id

        if self.root_node is None:
            self.root_node = decision_node
        else:
            parent.next_nodes[cur_class] = decision_node

        return decision_node

    def _build_tree_recursively(self, x: np.ndarray, y: np.ndarray, max_depth, cur_depth=0, parent=None, cur_class=1):
        root_entropy = calculate_entropy(y)
        x_trans = x.T
        info_gains = np.ndarray(shape=(len(x_trans),))
        cur_info_gain_ind = 0

        for _ in x_trans:
            _, _, y_left, y_right = split_data(x, y, feature_id=cur_info_gain_ind)

            info_gains[cur_info_gain_ind] = calculate_info_gain(root_entropy, y_left, y_right)

            cur_info_gain_ind += 1

        max_info_gain = max(info_gains)
        if max_info_gain == 0:
            best_feature_id = -1
        else:
            best_feature_id = np.where(info_gains == max_info_gain)[0][0]

        x_new_left, x_new_right, y_new_left, y_new_right = split_data(x, y, best_feature_id)

        if splitting_criteria_met(max_depth, cur_depth, y, best_feature_id, max_info_gain):
            self._create_leaf_node(parent, cur_class)

            if type(self.root_node) == LeafNode:
                if calculate_probability(y) >= 0.5:
                    self.root_node.prediction = 1
                else:
                    self.root_node.prediction = 0

        else:
            node = self._create_decision_node(parent, cur_class, best_feature_id)
            self._build_tree_recursively(x_new_left, y_new_left, max_depth, cur_depth+1, node, cur_class)

            cur_class = not cur_class

            self._build_tree_recursively(x_new_right, y_new_right, max_depth, cur_depth+1, node, cur_class)

    def fit(self, x: np.ndarray, y: np.ndarray, max_depth):
        self.input_shape = x.shape[1]
        if x.shape[0] != y.shape[0]:
            raise Exception("y must have the same size as x")

        self._build_tree_recursively(x, y, max_depth)

    def predict(self, x: np.ndarray):
        size = x.shape[0]
        predictions = np.ndarray(shape=(size,))

        cur_node = self.root_node

        if cur_node is None:
            print("Tree should be trained first")
            return

        cur_prediction = 0
        for test_input in x:
            cur_node = self.root_node
            while type(cur_node) != LeafNode and cur_node is not None:
                cur_node_feature = cur_node.feature_id

                feature_value = test_input[cur_node_feature]

                cur_node = cur_node.next_nodes[feature_value]

            if cur_node is None:
                raise Exception("Error occurred while training preventing the leaf node from predicting")

            predictions[cur_prediction] = cur_node.prediction

            cur_prediction += 1

        return predictions

    def write_tree(self, filename, input_features_names, output_names, cur_node, cur_depth=0):
        if type(cur_node) == LeafNode:
            cur_prediction = cur_node.prediction

            if output_names is None:
                output_name = str(bool(cur_prediction))
            else:
                output_name = output_names[cur_prediction]

            write_line(filename, output_name, cur_depth, None)

        else:
            cur_feature = cur_node.feature_id

            if input_features_names is None:
                feature_name = "feature[" + str(cur_feature) + "]"
            else:
                feature_name = input_features_names[cur_feature]

            write_line(filename, feature_name, cur_depth, 1)
            self.write_tree(filename, input_features_names, output_names, cur_node.next_nodes[1],
                            cur_depth+1)

            write_line(filename, feature_name, cur_depth, 0)
            self.write_tree(filename, input_features_names, output_names, cur_node.next_nodes[0],
                            cur_depth+1)

    def generate_pseudo_code(self, input_features_names=None, output_names=None, filename=None):
        if self.root_node is None:
            raise Exception("No Tree trained yet")

        if input_features_names is not None and len(input_features_names) != self.input_shape:
            raise Exception("Wrong input feature shape")

        else:
            if filename is not None:
                with open(filename, 'w') as file:
                    file.write("***************PSEUDO CODE***************\n\n")

            self.write_tree(
                filename=filename,
                input_features_names=input_features_names,
                output_names=output_names,
                cur_node=self.root_node
            )
