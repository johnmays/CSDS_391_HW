import data_tools
import tensorflow as tf
from tensorflow import keras


# re-importing the full dataset from the iris.csv file:
sepal_length, sepal_width, petal_length, petal_width, species = data_generator.iris_data_generator('data/irisdata.csv', full=True)
X = data_tools.create_data_vectors(sepal_length, sepal_width, petal_length, petal_width)
Y = data_tools.one_hot(species)

