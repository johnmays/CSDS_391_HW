import exercise_one
import exercise_two
import exercise_three
import numpy as np
import data_generator

m = np.array([0.6, 1.8])
b = -5.73
w = np.array([b, m[0], m[1]]).T

# importing the data from the iris.csv file:
sepal_length, sepal_width, petal_length, petal_width, species = data_generator.iris_data_generator('data/irisdata.csv')

# making a 2x100 matrix of the petal length and width data vectors:
X = data_generator.create_data_vectors(petal_length, petal_width)

# making a 3x100 augmented matrix with all ones in row 0, and petal length and width in rows 1 and 2
X_augmented = data_generator.create_augmented_data_vectors(petal_length, petal_width)

# HyperParameters
step_size = 0.0025


# Part 1:


def output_1c():
    exercise_one.plot_iris_data_with_decision_boundary(petal_length, petal_width, species, m=m, b=b)

def output_1d():
    exercise_one.surface_plot_input_space(m, b)


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


# Part 2:


def output_2b():
    mean_squared_error = exercise_two.mse(X, m, b, species)
    print(mean_squared_error)
    m_bad = np.array([1.0, 2.1])
    b_bad = -6
    exercise_two.plot_iris_data_with_two_decision_boundaries(petal_length, petal_width, species, m, b, m_bad, b_bad)
    mean_squared_error = exercise_two.mse(X, m_bad, b_bad, species)
    print(mean_squared_error)


def output_2e():
    w_new = np.copy(w)
    print(w_new)
    exercise_one.plot_iris_data_with_decision_boundary(petal_length, petal_width, species, subtitle="First", w=w_new)
    d_w = exercise_two.gradient_mse(X_augmented, w_new, species)
    # print(w_new.shape)
    # print(d_w.shape)
    w_new = w_new - step_size*d_w
    print(w_new)
    exercise_one.plot_iris_data_with_decision_boundary(petal_length, petal_width, species, subtitle="Second", w=w_new)


# Part 3:


def output_3a():
    iris_classifier = exercise_one.simple_classifier(w=w)
    print(iris_classifier.get_weights())
    w_final = exercise_three.fit(iris_classifier, step_size, X_augmented, petal_length, petal_width, species,
                                 progress_output=True)
    print(w_final)


def output_3c():
    w_random = exercise_three.random_weights()
    # w_random = np.array([2.44080189, 13.73806801, -2.08559572])
    print("Starting Vector:")
    print(w_random)
    print("MSE:")
    print(exercise_two.mse(X, w_random[1:3], w_random[0], species))
    iris_classifier = exercise_one.simple_classifier(w=w_random)
    print(iris_classifier.get_weights())
    w_final = exercise_three.fit(iris_classifier, step_size, X_augmented, petal_length, petal_width, species,
                                 progress_output=True)
    print(w_final)


# Put commands here:

# output_1c()

# output_1d()

# output_1e()

# output_2b()

# output_2e()

# output_3a()

# output_3c()




