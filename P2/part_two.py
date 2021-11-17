import part_one
import exceptions

import numpy as np


# Functions written in Part 2:


def mse(X, m, b, species):
    iris_classifer = part_one.simple_classifier(m, b)
    mse_sum = 0
    for i in range(len(X[0, :])):
        x_one = X[0, i]
        x_two = X[1, i]
        prediction = iris_classifer.classify(x_one, x_two)
        ground_truth = None
        if species[i] == "versicolor":
            ground_truth = 0
        elif species[i] == "virginica":
            ground_truth = 1
        else:
            raise exceptions.unexpected_class_error
        mse_sum += (ground_truth - prediction)**2

    return mse_sum/len(X[0, :])
