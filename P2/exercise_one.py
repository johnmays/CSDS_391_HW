import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Functions written in Part 1:


def plot_iris_data(petal_length, petal_width, species):
    """For 1a.  Simply plots versicolor and virginica with two colors"""
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

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.scatter(versicolor_petal_length, versicolor_petal_width, color='indigo', alpha=0.5, label='Versicolor')
    ax.scatter(virginica_petal_length, virginica_petal_width, color='orchid', alpha=0.5, label='Virginica')
    plt.title("Iris Data")
    plt.ylabel("Petal Width (cm)")
    plt.xlabel("Petal Length (cm)")
    plt.xlim(0, 7.5)
    plt.legend()
    plt.show()


class decision_boundary:
    """Helper function to plot the decision boundary"""
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def get_x_two(self, x_one):
        x_two = -(self.m[0]/self.m[1])*x_one-(self.b/self.m[1])
        return x_two


def plot_iris_data_with_decision_boundary(petal_length, petal_width, species, m, b):
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
    # drawing the line
    x_ones = np.linspace(0, 7.5, 75)
    x_twos = []
    iris_decision_boundary = decision_boundary(m, b)
    for x_one in x_ones:
        x_twos.append(iris_decision_boundary.get_x_two(x_one))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.scatter(versicolor_petal_length, versicolor_petal_width, color='indigo', alpha=0.5, label='Versicolor')
    ax.scatter(virginica_petal_length, virginica_petal_width, color='orchid', alpha=0.5, label='Virginica')
    plt.plot(x_ones, x_twos, color='black')
    plt.fill_between(x_ones, x_twos, np.max(x_twos), color='orchid', alpha=0.1)
    plt.fill_between(x_ones, x_twos, color='indigo', alpha=0.15)
    plt.title("Iris Data")
    plt.ylabel("Petal Width (cm) [x\u2082]")
    plt.xlabel("Petal Length (cm) [x\u2081]")
    plt.xlim(0, 7.5)
    plt.ylim(0.8, 2.6)
    plt.legend()
    plt.show()


class simple_classifier:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def classify(self, x_one, x_two):
        y = self.m[0]*x_one + self.m[1]*x_two + self.b
        sigmoid = 1 / (1 + np.exp(-y))
        if sigmoid <= 0.5:  # 2nd iris class: Versicolor
            return 0
        else:  # 3rd iris class: Virginica
            return 1


def surface_plot_input_space(m, b):
    x_one = np.linspace(0, 7, num=70+1)  # Petal Length
    x_two = np.linspace(0, 2.6, num=26+1)  # Petal Width

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x_one, x_two = np.meshgrid(x_one, x_two)
    y = m[0]*x_one+m[1]*x_two + b
    sigmoid = 1 / (1 + np.exp(-y))
    surf = ax.plot_surface(x_one, x_two, sigmoid, cmap=mpl.cm.RdPu, linewidth=0, antialiased=False)
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 2.6)
    ax.set_zlim(0, 1.0)
    ax.set_xlabel("Petal Length (cm) [x\u2081]")
    ax.set_ylabel("Petal Width (cm) [x\u2082]")
    ax.set_zlabel("Sigmoid Value")
    ax.view_init(elev=15., azim=-50)

    plt.show()


def test_simple_classifier(index, petal_length, petal_width, species, m, b):
    iris_classifier = simple_classifier(m, b)
    classifier_output = iris_classifier.classify(petal_length[index], petal_width[index])
    classifier_class = ""
    if classifier_output == 0:
        classifier_class = "versicolor"
    else:
        classifier_class = "virginica"

    print("Petal Length:", petal_length[index], ", Petal Width:", petal_width[index], ", True Class:", species[index], ", Simple Classifier Output:", classifier_output, "({cc})".format(cc=classifier_class))


def plot_select_iris_data_1e(petal_length, petal_width, species, m, b):
    versicolor_petal_length = []
    versicolor_petal_width = []
    virginica_petal_length = []
    virginica_petal_width = []
    indices = [50, 60, 80, 100, 120, 140, 106, 119, 83]
    for index in indices:
        if species[index] == 'versicolor':
            versicolor_petal_length.append(petal_length[index])
            versicolor_petal_width.append(petal_width[index])
        elif species[index] == 'virginica':
            virginica_petal_length.append(petal_length[index])
            virginica_petal_width.append(petal_width[index])

    x_ones = np.linspace(0, 7.5, 75)
    x_twos = []
    iris_decision_boundary = decision_boundary(m, b)
    for x_one in x_ones:
        x_twos.append(iris_decision_boundary.get_x_two(x_one))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.scatter(versicolor_petal_length, versicolor_petal_width, color='indigo', alpha=0.5, label='Versicolor')
    ax.scatter(virginica_petal_length, virginica_petal_width, color='orchid', alpha=0.5, label='Virginica')
    plt.plot(x_ones, x_twos, color='black')
    # plt.fill_between(x_ones, x_twos, np.max(x_twos), color='orchid', alpha=0.1)
    # plt.fill_between(x_ones, x_twos, color='indigo', alpha=0.15)
    plt.title("Nine Select Points of Iris Data")
    plt.ylabel("Petal Width (cm)")
    plt.xlabel("Petal Length (cm)")
    plt.xlim(0, 7.5)
    plt.ylim(0.8, 2.6)
    plt.legend()
    plt.show()


