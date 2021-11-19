import csv
import numpy as np


def iris_data_generator(data_path):
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
            if row[4] != "setosa":
                sepal_length.append(float(row[0]))
                sepal_width.append(float(row[1]))
                petal_length.append(float(row[2]))
                petal_width.append(float(row[3]))
                species.append(row[4])

    return sepal_length, sepal_width, petal_length, petal_width, species


def create_data_vectors(x_one, x_two):
    X = np.zeros((2, len(x_one)))
    for i in range(len(x_one)):
        X[0, i] = x_one[i]
        X[1, i] = x_two[i]
    return X  # a 2x(length) matrix with columns individual x vectors as columns

def create_augmented_data_vectors(x_one, x_two):
    X_augmented = np.zeros((3, len(x_one)))
    for i in range(len(x_one)):
        X_augmented[0, i] = 1
        X_augmented[1, i] = x_one[i]
        X_augmented[2, i] = x_two[i]
    return X_augmented  # a 2x(length) matrix with columns individual x vectors as columns

# def create_data_vectors(x_one, x_two, x_three, x_four):
