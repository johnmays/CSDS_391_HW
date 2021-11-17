import part_one
import part_two
import numpy as np
import data_generator

m = np.array([0.6, 1.8])
b = -5.73

# importing the data from the iris.csv file:
sepal_length, sepal_width, petal_length, petal_width, species = data_generator.iris_data_generator('data/irisdata.csv')

# making a 2x100 matrix of the petal length and width data vectors:
X = data_generator.create_data_vectors(petal_length, petal_width)

# Part 1:


def output_1e():

    print("Unambiguously Versicolor:")
    part_one.test_simple_classifier(0, petal_length, petal_width, species)
    part_one.test_simple_classifier(10, petal_length, petal_width, species)
    part_one.test_simple_classifier(30, petal_length, petal_width, species)
    print("Unambiguously Virginica:")
    part_one.test_simple_classifier(50, petal_length, petal_width, species)
    part_one.test_simple_classifier(70, petal_length, petal_width, species)
    part_one.test_simple_classifier(90, petal_length, petal_width, species)
    print("Near the Decision Boundary:")
    part_one.test_simple_classifier(56, petal_length, petal_width, species)
    part_one.test_simple_classifier(69, petal_length, petal_width, species)
    part_one.test_simple_classifier(33, petal_length, petal_width, species)

    part_one.plot_select_iris_data_1e(petal_length, petal_width, species)


# Part 2:

def output_2b():
    mean_squared_error = part_two.mse(X, m, b, species)
    print(mean_squared_error)
    m_bad = np.array([1.0, 2.1])
    b_bad = -6
    part_two.plot_iris_data_with_two_decision_boundaries(petal_length, petal_width, species, m, b, m_bad, b_bad)
    mean_squared_error = part_two.mse(X, m_bad, b_bad, species)
    print(mean_squared_error)

