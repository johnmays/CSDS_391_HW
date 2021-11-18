import exercise_one
import exercise_two
import numpy as np
import data_generator

m = np.array([0.6, 1.8])
b = -5  # b = -5.73
w = np.array([m[0], m[1], b]).T

# importing the data from the iris.csv file:
sepal_length, sepal_width, petal_length, petal_width, species = data_generator.iris_data_generator('data/irisdata.csv')

# making a 2x100 matrix of the petal length and width data vectors:
X = data_generator.create_data_vectors(petal_length, petal_width)

# making a 3x100 augmented matrix with all ones in row 0, and petal length and width in rows 1 and 2
X_augmented = data_generator.create_augmented_data_vectors(petal_length, petal_width)

# HyperParameters
step_size = 0.05


# Part 1:


# 1d:

# exercise_one.surface_plot_input_space(m, b)

# 1e:


def output_1e():

    print("Unambiguously Versicolor:")
    exercise_one.test_simple_classifier(0, petal_length, petal_width, species, m, b)
    exercise_one.test_simple_classifier(10, petal_length, petal_width, species, m, b)
    exercise_one.test_simple_classifier(30, petal_length, petal_width, species, m, b)
    print("Unambiguously Virginica:")
    exercise_one.test_simple_classifier(50, petal_length, petal_width, species, m, b)
    exercise_one.test_simple_classifier(70, petal_length, petal_width, species, m, b)
    exercise_one.test_simple_classifier(90, petal_length, petal_width, species, m, b)
    print("Near the Decision Boundary:")
    exercise_one.test_simple_classifier(56, petal_length, petal_width, species, m, b)
    exercise_one.test_simple_classifier(69, petal_length, petal_width, species, m, b)
    exercise_one.test_simple_classifier(33, petal_length, petal_width, species, m, b)

    exercise_one.plot_select_iris_data_1e(petal_length, petal_width, species, m, b)


# output_1e()


# Part 2:


def output_2b():
    mean_squared_error = exercise_two.mse(X, m, b, species)
    print(mean_squared_error)
    m_bad = np.array([1.0, 2.1])
    b_bad = -6
    exercise_two.plot_iris_data_with_two_decision_boundaries(petal_length, petal_width, species, m, b, m_bad, b_bad)
    mean_squared_error = exercise_two.mse(X, m_bad, b_bad, species)
    print(mean_squared_error)


# 2e:

w_new = np.copy(w)
print(w_new)
exercise_one.plot_iris_data_with_decision_boundary(petal_length, petal_width, species, w_new[0:2], w_new[2])
d_w = exercise_two.gradient_mse(X_augmented, w_new, species)
# print(w_new.shape)
# print(d_w.shape)
w_new = w_new - step_size*d_w
print(w_new)
exercise_one.plot_iris_data_with_decision_boundary(petal_length, petal_width, species, w_new[0:2], w_new[2])





