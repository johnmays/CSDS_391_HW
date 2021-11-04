import matplotlib.pyplot as plt
import numpy as np
import math
import random


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


def plot_2_a(bag):
    d = []
    num_observations = 100
    if bag == "h1":
        prior = 0.1
        for i in range(num_observations):
            d.append('c')
    elif bag == "h2":
        for i in range(num_observations):
            if random.random() < 0.75:
                d.append('c')
            else:
                d.append('l')
    elif bag == "h3":
        for i in range(num_observations):
            if random.random() < 0.50:
                d.append('c')
            else:
                d.append('l')
    elif bag == "h4":
        for i in range(num_observations):
            if random.random() < 0.25:
                d.append('c')
            else:
                d.append('l')
    elif bag == "h5":
        for i in range(num_observations):
            d.append('l')
    else:
        print("Invalid argument. Needs arguments h1, h2, h3, or h4.")
        return

    priors = [0.1, 0.2, 0.4, 0.2, 0.1]
    posteriors_h1 = [priors[0]]
    posteriors_h2 = [priors[1]]
    posteriors_h3 = [priors[2]]
    posteriors_h4 = [priors[3]]
    posteriors_h5 = [priors[4]]
    likelihoods_h1 = 1
    likelihoods_h2 = 1
    likelihoods_h3 = 1
    likelihoods_h4 = 1
    likelihoods_h5 = 1

    for observation in d:
        likelihoods_h1 = likelihoods_h1 * h1(observation)
        likelihoods_h2 = likelihoods_h2 * h2(observation)
        likelihoods_h3 = likelihoods_h3 * h3(observation)
        likelihoods_h4 = likelihoods_h4 * h4(observation)
        likelihoods_h5 = likelihoods_h5 * h5(observation)

        # create posteriors vector for each hypothesis:
        posteriors_vector = np.array([likelihoods_h1 * priors[0], likelihoods_h2 * priors[1], likelihoods_h3 * priors[2], likelihoods_h4 * priors[3], likelihoods_h5 * priors[4]])
        # normalize it:
        posteriors_vector = posteriors_vector / sum(posteriors_vector)

        posteriors_h1.append(posteriors_vector[0])
        posteriors_h2.append(posteriors_vector[1])
        posteriors_h3.append(posteriors_vector[2])
        posteriors_h4.append(posteriors_vector[3])
        posteriors_h5.append(posteriors_vector[4])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.spines['bottom'].set_position('zero')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # ax.set_yticklabels([])
    plt.plot(np.linspace(0, len(d), len(d)+1), posteriors_h1, 'tab:orange', label='p(h1|d)')
    plt.plot(np.linspace(0, len(d), len(d) + 1), posteriors_h2, 'tab:green', label='p(h2|d)')
    plt.plot(np.linspace(0, len(d), len(d) + 1), posteriors_h3, 'tab:blue', label='p(h3|d)')
    plt.plot(np.linspace(0, len(d), len(d) + 1), posteriors_h4, 'tab:grey', label='p(h4|d)')
    plt.plot(np.linspace(0, len(d), len(d) + 1), posteriors_h5, 'tab:brown', label='p(h5|d)')
    plt.title("Posteriors of h1,h2,h3,h4, and h5 given data from {} bag".format(bag))
    plt.ylabel("Posterior probability of hypothesis")
    plt.xlabel("Number of observations in d vector")
    plt.legend()
    plt.show()

    probabilities_next_is_lime = []

    for i in range(len(posteriors_h1)):
        probability_next_is_lime = 0
        probability_next_is_lime += h1('l') * posteriors_h1[i]
        probability_next_is_lime += h2('l') * posteriors_h2[i]
        probability_next_is_lime += h3('l') * posteriors_h3[i]
        probability_next_is_lime += h4('l') * posteriors_h4[i]
        probability_next_is_lime += h5('l') * posteriors_h5[i]

        probabilities_next_is_lime.append(probability_next_is_lime)

    plt.clf()
    plt.plot(np.linspace(0, len(d), len(d) + 1), probabilities_next_is_lime, 'tab:brown', label='p(h5|d)')
    plt.title("Probability that the next candy is lime given data from {} bag".format(bag))
    plt.ylabel("Probability that the next candy is lime")
    plt.xlabel("Number of observations in d vector")
    plt.show()


def plot_2_c(bag):
    priors = [0.1, 0.2, 0.4, 0.2, 0.1]

    num_trials = 2500  # rows
    num_observations = 100  # columns
    d = np.empty((num_trials, num_observations), dtype=str)

    posteriors_h1 = np.empty((num_trials, num_observations+1), dtype=float)
    posteriors_h2 = np.empty((num_trials, num_observations + 1), dtype=float)
    posteriors_h3 = np.empty((num_trials, num_observations + 1), dtype=float)
    posteriors_h4 = np.empty((num_trials, num_observations + 1), dtype=float)
    posteriors_h5 = np.empty((num_trials, num_observations + 1), dtype=float)

    for i in range(num_trials):
        for j in range(num_observations):
            if bag == "h1":
                d[i, j] = 'c'
            elif bag == "h2":
                if random.random() < 0.75:
                    d[i, j] = 'c'
                else:
                    d[i, j] = 'l'
            elif bag == "h3":
                if random.random() < 0.50:
                    d[i, j] = 'c'
                else:
                    d[i, j] = 'l'
            elif bag == "h4":
                if random.random() < 0.25:
                    d[i, j] = 'c'
                else:
                    d[i, j] = 'l'
            elif bag == "h5":
                d[i, j] = 'l'
            else:
                print("Invalid argument. Needs arguments h1, h2, h3, h4, or h5.")
                return

        posteriors_h1[i, 0] = priors[0]
        posteriors_h2[i, 0] = priors[1]
        posteriors_h3[i, 0] = priors[2]
        posteriors_h4[i, 0] = priors[3]
        posteriors_h5[i, 0] = priors[4]
        likelihoods_h1 = 1
        likelihoods_h2 = 1
        likelihoods_h3 = 1
        likelihoods_h4 = 1
        likelihoods_h5 = 1

        for k in range(len(d[i])):
            observation = d[i,k]
            likelihoods_h1 = likelihoods_h1 * h1(observation)
            likelihoods_h2 = likelihoods_h2 * h2(observation)
            likelihoods_h3 = likelihoods_h3 * h3(observation)
            likelihoods_h4 = likelihoods_h4 * h4(observation)
            likelihoods_h5 = likelihoods_h5 * h5(observation)

            # create posteriors vector for each hypothesis:
            posteriors_vector = np.array(
                [likelihoods_h1 * priors[0], likelihoods_h2 * priors[1], likelihoods_h3 * priors[2],
                 likelihoods_h4 * priors[3], likelihoods_h5 * priors[4]])
            # normalize it:
            posteriors_vector = posteriors_vector / sum(posteriors_vector)

            posteriors_h1[i, k + 1] = posteriors_vector[0]
            posteriors_h2[i, k + 1] = posteriors_vector[1]
            posteriors_h3[i, k + 1] = posteriors_vector[2]
            posteriors_h4[i, k + 1] = posteriors_vector[3]
            posteriors_h5[i, k + 1] = posteriors_vector[4]

    averages_h1 = []
    averages_h2 = []
    averages_h3 = []
    averages_h4 = []
    averages_h5 = []
    for i in range(len(posteriors_h1[0])):
        averages_h1.append(np.average(posteriors_h1[:, i]))
        averages_h2.append(np.average(posteriors_h2[:, i]))
        averages_h3.append(np.average(posteriors_h3[:, i]))
        averages_h4.append(np.average(posteriors_h4[:, i]))
        averages_h5.append(np.average(posteriors_h5[:, i]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_ylim([0.00, 0.35])
    # ax.spines['bottom'].set_position('zero')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # ax.set_yticklabels([])
    plt.plot(np.linspace(0, len(d[0]), len(d[0]) + 1), averages_h1, 'tab:orange', label='p(h1|d)')
    plt.plot(np.linspace(0, len(d[0]), len(d[0]) + 1), averages_h2, 'tab:green', label='p(h2|d)')
    plt.plot(np.linspace(0, len(d[0]), len(d[0]) + 1), averages_h3, 'tab:blue', label='p(h3|d)')
    plt.plot(np.linspace(0, len(d[0]), len(d[0]) + 1), averages_h4, 'tab:grey', label='p(h4|d)')
    plt.plot(np.linspace(0, len(d[0]), len(d[0]) + 1), averages_h5, 'tab:brown', label='p(h5|d)')
    plt.title("Average Posteriors of Hypotheses given data from {} bag".format(bag))
    plt.ylabel("Average posterior probability of hypothesis")
    plt.xlabel("Number of observations in d vector")
    plt.legend()
    plt.show()


def h1(flavor):
    """returns likelihood of hypothesis, given an individual flavor value"""
    if flavor == 'c':
        return 1.0
    elif flavor == 'l':
        return 0.0
    else:
        print("Unexpected flavor value")
        return


def h2(flavor):
    """returns likelihood of hypothesis, given an individual flavor value"""
    if flavor == 'c':
        return 0.75
    elif flavor == 'l':
        return 0.25
    else:
        print("Unexpected flavor value")
        return


def h3(flavor):
    """returns likelihood of hypothesis, given an individual flavor value"""
    if flavor == 'c':
        return 0.5
    elif flavor == 'l':
        return 0.5
    else:
        print("Unexpected flavor value")
        return


def h4(flavor):
    """returns likelihood of hypothesis, given an individual flavor value"""
    if flavor == 'c':
        return 0.25
    elif flavor == 'l':
        return 0.75
    else:
        print("Unexpected flavor value")
        return


def h5(flavor):
    """returns likelihood of hypothesis, given an individual flavor value"""
    if flavor == 'c':
        return 0.0
    elif flavor == 'l':
        return 1.0
    else:
        print("Unexpected flavor value")
        return


# plot_1_b()

# plot_1_c(1, 1)
# plot_1_c(2, 2)
# plot_1_c(2, 3)
# plot_1_c(3, 4)

# plot_2_a("h1")
# plot_2_a("h2")
# plot_2_a("h3")
# plot_2_a("h4")
# plot_2_a("h5")

plot_2_c("h1")
plot_2_c("h2")
plot_2_c("h3")
plot_2_c("h4")
plot_2_c("h5")
