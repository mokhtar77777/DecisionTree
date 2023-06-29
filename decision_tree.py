from data_splitter import *
from tree_formator import *


class DecisionNode:
    def __init__(self):
        """
        Each instance created from this node will be responsible for predicting certain feature
        Each Node will be pointing to two nodes. Each node can be either another DecisionNode or a LeafNode
        """
        self.next_nodes = [None] * 2
        self.feature_id = -1


class LeafNode:
    def __init__(self):
        """
        Each instance created from this node will be responsible for a certain prediction
        Unlike DecisionNode, LeafNodes do not point to any nodes
        """
        self.prediction = None


class DecisionTreeClassifier:
    def __init__(self):
        self.root_node = None
        self.input_shape = -1

    def _create_leaf_node(self, parent: DecisionNode, cur_class) -> None:
        """
        This private method creates a new LeafNode with prediction cur_class and allows the parent DecisionNode
        to point to the LeafNode which will be created
        :param parent: DecisionNode pointing to the new LeafNode. If None, root_node will be the new LeafNode.
        :param cur_class: Prediction of the LeafNode which will be created (1 or 0)
        """
        leaf_node = LeafNode()
        leaf_node.prediction = cur_class
        if parent is None:
            self.root_node = leaf_node
        else:
            parent.next_nodes[cur_class] = leaf_node

    def _create_decision_node(self, parent: DecisionNode, cur_class, feature_id) -> DecisionNode:
        """
        This private method creates a new DecisionNode with feature feature_id and allows the parent DecisionNode
        to point to the DecisionNode which will be created
        :param parent: DecisionNode pointing to the new DecisionNode. If None, root_node will be the new DecisionNode.
        :param cur_class: Passed as (1 or 0) corresponding to which branch the new DecisionNode will be at
        :param feature_id: Feature number the new DecisionNode will be responsible for
        :return: New DecisionNode
        """
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
        """
        This public method is responsible for training and building the decision tree.
        Warning: Decision Trees are very sensitive to changes in training examples.
        Note: Neither tree ensemble nor random forest algorithm is built implicitly in this method
        :param x: Matrix (2D array) where each row corresponds to a training example input.
        - All examples must have the values of 0 or 1
        - All rows must have the same size.
        - Each column corresponds to a certain feature.
        - The first column has feature_id of 0, the second has feature_id of 1, and so on..
        :param y: 1D array (row vector) where each element corresponds to a training label
        - All labels must have the values of 0 or 1
        - Number of labels must be equal to number of training examples inputs
        :param max_depth: Max Depth the tree can reach. Root Node is at depth of 0
        """
        self.input_shape = x.shape[1]
        if x.shape[0] != y.shape[0]:
            raise Exception("y must have the same size as x")

        self._build_tree_recursively(x, y, max_depth)

    def predict(self, x: np.ndarray):
        """
        This public method is responsible for prediction. This function can be called only after called the "fit" method
        :param x: Matrix (2D array) where each corresponds to a test example input
        - All examples must have the values of 0 or 1
        - All rows must have the same size.
        - Each column corresponds to a certain feature.
        - The first column has feature_id of 0, the second has feature_id of 1, and so on..
        :return: 1D array (row vector) where each element corresponds to a certain prediction
        """
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

    def _write_tree(self, filename, input_features_names, output_names, cur_node, cur_depth=0):
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
            self._write_tree(filename, input_features_names, output_names, cur_node.next_nodes[1], cur_depth+1)

            write_line(filename, feature_name, cur_depth, 0)
            self._write_tree(filename, input_features_names, output_names, cur_node.next_nodes[0], cur_depth+1)

    def generate_pseudo_code(self, input_features_names=None, output_names=None, filename=None):
        """
        This public method generated a simple if else pseudo code to have a better interpretation about what the
        decision tree is actually doing. This method can only be called after called the method "fit". You can choose
        whether the pseudo code to be saved in a text file or only printed in the terminal
        :param input_features_names: A list of strings where each string corresponds to the name of the feature
        Example: ["ear_shape_is_pointy", "face_shape_is_round", "whiskers_present"]
        Feature 0 name: ear_shape_is_pointy
        Feature 1 name: face_shape_is_round
        Feature 2 name: whiskers_present
        If None, instead of using any strings, it will use "feature[feature_id]"
        :param output_names: A list of 2 strings where each string corresponds to the name of output
        Example: ["not_cat", "cat"]
        False Output: not_cat
        True Output: cat
        If None, instead of using any strings, it will use "True" or "False"
        :param filename: Text filename. If None, the pseudo code will be printed in the console
        """
        if self.root_node is None:
            raise Exception("No Tree trained yet")

        if input_features_names is not None and len(input_features_names) != self.input_shape:
            raise Exception("Wrong input feature shape")

        else:
            if filename is not None:
                with open(filename, 'w') as file:
                    file.write("***************PSEUDO CODE***************\n\n")

            self._write_tree(
                filename=filename,
                input_features_names=input_features_names,
                output_names=output_names,
                cur_node=self.root_node
            )
