import data_tools
import tensorflow as tf
from tensorflow import keras


# re-importing the full dataset from the iris.csv file:
sepal_length, sepal_width, petal_length, petal_width, species = data_tools.iris_data_generator('data/irisdata.csv', full=True)
X = data_tools.create_data_vectors(sepal_length, sepal_width, petal_length, petal_width)
Y = data_tools.one_hot(species)
X_test, Y_test, X_train, Y_train = data_tools.data_split(X, Y, split_percent=0.2)

model = keras.models.Sequential()
model.add(keras.layers.Dense(4, input_shape=(X[0].size, ), activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, epochs=25)

# print(type(Y_train))
