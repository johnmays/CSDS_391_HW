import csv


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
            sepal_length.append(float(row[0]))
            sepal_width.append(float(row[1]))
            petal_length.append(float(row[2]))
            petal_width.append(float(row[3]))
            species.append(row[4])

    return sepal_length, sepal_width, petal_length, petal_width, species
