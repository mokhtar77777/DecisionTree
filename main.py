from decision_tree import DecisionTreeClassifier
import numpy as np

tree = DecisionTreeClassifier()

x = np.array(
    [
        [1, 1, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]
)

y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

tree.fit(x, y, max_depth=100)

y_predict = tree.predict(x)

print(y_predict)

tree.generate_pseudo_code(
    input_features_names=("ear_shape_is_pointy", "face_shape_is_round", "whiskers_present"),
    output_names=("not_cat", "cat"),
    filename="pseudo.txt"
)
