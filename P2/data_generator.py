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


def create_data_vectors(X_one, X_two):
    X = np.zeros((2, len(X_one)))
    for i in range(len(X_one)):
        X[0, i] = X_one[i]
        X[1, i] = X_two[i]
    return X  # a 2x(length) matrix with columns individual x vectors as columns

# def create_data_vectors(x_one, x_two, x_three, x_four):