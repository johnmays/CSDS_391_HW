import matplotlib.pyplot as plt
import numpy as np
import math


def plot_1_b():
    # 1a:
    n = 4  # number of trials
    theta = 0.75  # probability of heads (1) per turn

    y_values = []
    p_values = []

    y = 0
    while y <= 4:
        y_values.append(y)
        comb = np.math.factorial(n) / (np.math.factorial(n - y) * np.math.factorial(y))
        p = comb * (math.pow(theta, y)) * (math.pow((1 - theta), (n - y)))
        p_values.append(p)
        y += 1
    plt.bar(y_values, p_values)
    plt.title("Likelihood for n = 4 and Θ = 0.75")
    plt.ylabel("p(y|n=4, Θ=0.75)")
    plt.xlabel("y")
    plt.show()


def plot_1_c(y, n):
    thetas = np.linspace(0, 1, 100)

    comb = np.math.factorial(n) / (np.math.factorial(n - y) * np.math.factorial(y))
    likelihoods = []
    for theta in thetas:
        likelihoods.append(comb * (math.pow(theta, y)) * (math.pow((1 - theta), (n - y))))

    # priors = np.copy(thetas)  # prior for y = 1, n = 1

    posteriors = []
    for likelihood in likelihoods:
        posteriors.append(likelihood * (1 + n))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['bottom'].set_position('zero')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticklabels([])
    plt.plot(thetas, posteriors, 'b')
    plt.title("Posterior for Coin Tosses")
    plt.ylabel("p(Θ|y={}, n={})".format(y, n))
    plt.xlabel("Θ")
    plt.show()


# plot_1_b()

# plot_1_c(1, 1)
# plot_1_c(2, 2)
# plot_1_c(2, 3)
# plot_1_c(3, 4)
