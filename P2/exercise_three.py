import exercise_one
import exercise_two

import numpy as np
import matplotlib.pyplot as plt


def fit(classifier, step_size, X_augmented, petal_length, petal_width, species, output=False):
    threshold = 0.5
    gradient = 0
    previous_gradient = 0

    # Parameters for Graphing/Output:
    num_iterations = 0
    weights_store = []
    mse_store = []
    while True:
        num_iterations += 1  # for graphing
        gradient = exercise_two.gradient_mse(X_augmented, classifier.get_weights(), species)
        new_weights = classifier.get_weights()-step_size*gradient
        if output:
            mse_store.append(exercise_two.mse(X_augmented[1:, :], new_weights[1:3], new_weights[0], species))  # for graphing
        weights_store.append(new_weights)  # for graphing
        classifier.set_weights(new_weights)
        if np.linalg.norm(gradient) < threshold:  # convergence condition
            break
        previous_gradient = gradient
    print(num_iterations)
    if output:  # is true
        plot_loss_over_iterations(mse_store, num_iterations)
        plot_decision_boundaries_over_iterations(petal_length, petal_width, species, weights_store, num_iterations, skip_size=2)

    return classifier.get_weights()

def plot_loss_over_iterations(mse_store, num_iterations):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(range(1, num_iterations+1), mse_store, color='indigo')
    plt.title("Mean Squared Error (Loss) vs. Num. Iterations")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Number of Iterations")
    # plt.xlim(0, 7.5)
    plt.show()


def plot_decision_boundaries_over_iterations(petal_length, petal_width, species, weights_store, num_iterations, skip_size=50):
    # Creating two datasets to plot
    versicolor_petal_length = []
    versicolor_petal_width = []
    virginica_petal_length = []
    virginica_petal_width = []
    for l, w, s in zip(petal_length, petal_width, species):
        if s == 'versicolor':
            versicolor_petal_length.append(l)
            versicolor_petal_width.append(w)
        elif s == 'virginica':
            virginica_petal_length.append(l)
            virginica_petal_width.append(w)

    # Setting up figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # Plotting data points:
    ax.scatter(versicolor_petal_length, versicolor_petal_width, color='indigo', alpha=0.5, label='Versicolor')
    ax.scatter(virginica_petal_length, virginica_petal_width, color='orchid', alpha=0.5, label='Virginica')
    # Drawing Multiple Decision Boundaries:
    for i in range(len(weights_store)):
        current_weights_vector = weights_store[i]
        if i % skip_size == 0:
            x_ones = np.linspace(0, 7.5, 75)
            x_twos = []
            iris_decision_boundary = exercise_one.decision_boundary(w=current_weights_vector)
            for x_one in x_ones:
                x_twos.append(iris_decision_boundary.get_x_two(x_one))
            plt.plot(x_ones, x_twos, color='black', alpha=0.05)
    # Labeling:
    plt.title("Iris Data")
    plt.ylabel("Petal Width (cm) [x\u2082]")
    plt.xlabel("Petal Length (cm) [x\u2081]")
    plt.xlim(0, 7.5)
    plt.ylim(0.8, 2.6)
    plt.legend()

    plt.show()
