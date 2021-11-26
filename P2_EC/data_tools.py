import exceptions
import csv
import numpy as np
import random
import math


def iris_data_generator(data_path, full=False):
    iris_data_file = open(data_path)
    iris_reader = csv.reader(iris_data_file)
    # iris_data = list(iris_reader)

    sepal_length = []
    sepal_width = []
    petal_length = []
    petal_width = []
    species = []

    for row in iris_reader:
        if iris_reader.line_num != 1:
            if not full:
                if row[4] != "setosa":  # if full is not specified as true, exclude Setosa
                    sepal_length.append(float(row[0]))
                    sepal_width.append(float(row[1]))
                    petal_length.append(float(row[2]))
                    petal_width.append(float(row[3]))
                    species.append(row[4])
            else:
                sepal_length.append(float(row[0]))
                sepal_width.append(float(row[1]))
                petal_length.append(float(row[2]))
                petal_width.append(float(row[3]))
                species.append(row[4])

    return sepal_length, sepal_width, petal_length, petal_width, species


def create_data_vectors(*args):
    if len(args) == 2:
        x_one = args[0]
        x_two = args[1]
        X = np.zeros((len(x_one), 2))
        for i in range(len(x_one)):
            X[i, 0] = x_one[i]
            X[i, 1] = x_two[i]
    elif len(args) == 4:
        x_one = args[0]
        x_two = args[1]
        x_three = args[2]
        x_four = args[3]
        X = np.zeros((len(x_one), 4))
        for i in range(len(x_one)):
            X[i, 0] = x_one[i]
            X[i, 1] = x_two[i]
            X[i, 2] = x_three[i]
            X[i, 3] = x_four[i]
    else:
        raise exceptions.insufficient_arguments_error
    return X  # a 2x(length) matrix with columns individual x vectors as columns


def create_augmented_data_vectors(x_one, x_two):
    X_augmented = np.zeros((3, len(x_one)))
    for i in range(len(x_one)):
        X_augmented[0, i] = 1
        X_augmented[1, i] = x_one[i]
        X_augmented[2, i] = x_two[i]
    return X_augmented  # a 2x(length) matrix with columns individual x vectors as columns


def one_hot(category_vector):
    categories = []
    for entry in category_vector:
        if entry not in categories:
            categories.append(entry)
    Y = []
    for entry in category_vector:
        one_hot_vector = []
        for category in categories:
            if entry == category:
                one_hot_vector.append(1)
            else:
                one_hot_vector.append(0)
        Y.append(one_hot_vector)
    Y = np.asarray(Y)
    return Y


def data_split(X, Y, split_percent=0.5):
    """Split one will contain split_percent*100% of the data in random order"""
    X_split_one = []
    Y_split_one = []
    X_split_two = []
    Y_split_two = []

    data_indices = []
    for index in range(len(Y)):  # creates a list of all of the indices of entries in the dataset
        data_indices.append(index)
    # Creating the first split based upon random selections:
    while len(data_indices) > (1-split_percent)*len(Y):
        index = data_indices.pop(math.floor(random.random()*len(data_indices)))
        X_split_one.append(X[index])
        Y_split_one.append(Y[index])
    # Creating the second split based upon what's left:
    for index in data_indices:
        X_split_two.append(X[index])
        Y_split_two.append(Y[index])
    X_split_one = np.asarray(X_split_one)
    X_split_two = np.asarray(X_split_two)
    Y_split_one = np.asarray(Y_split_one)
    Y_split_two = np.asarray(Y_split_two)
    return X_split_one, Y_split_one, X_split_two, Y_split_two


