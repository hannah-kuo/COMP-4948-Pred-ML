""" Configure the optimal learning rate, momentum value and total number of nodes in a single layer network """

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf

# load the dataset
PATH = "C:/PredML/"
df = pd.read_csv(PATH + 'fluDiagnosis.csv')
# split into input (X) and output (y) variables
print(df)

X = df[['A', 'B']]
y = df[['Diagnosed']]
# Split into train and test data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


def buildModel(num_nodes):
    # define the keras model
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=2, activation='relu',
                    kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))

    # opitimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, name="SGD")
    # opitimizer = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, name="SGD")
    opitimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.7, name="SGD")

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer=opitimizer, metrics=['accuracy'])

    # fit the keras model on the dataset
    # history = model.fit(X, y, epochs=80, batch_size=10, validation_data=(X_test, y_test))
    history = model.fit(X_train, y_train, epochs=80, batch_size=10, validation_data=(X_test, y_test))
    # evaluate the keras model

    # Evaluate the model.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: ' + str(acc) + ' Num nodes: ' + str(num_nodes))
    return history


def showLoss(history, numNodes):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history for training data.
    actualLabel = str(numNodes) + " nodes"
    plt.subplot(1, 2, 1)
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actualLabel)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()


def showAccuracy(history, numNodes):
    # Get training and test loss histories
    training_loss = history.history['accuracy']
    validation_loss = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)

    actualLabel = str(numNodes) + " nodes"
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actualLabel)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()


# nodeCounts = [170, 200, 230]
# nodeCounts = [10, 50, 100, 150, 200, 250]
# nodeCounts = [50, 150, 200]

# nodeCounts = [150, 180, 200]
# nodeCounts = [3, 6, 9]
# nodeCounts = [50, 100, 150, 250]
nodeCounts = [150, 170, 200]

plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
plt.title("Exercise 2")

for i in range(0, len(nodeCounts)):
    history = buildModel(nodeCounts[i])
    showLoss(history, nodeCounts[i])
    showAccuracy(history, nodeCounts[i])

plt.show()
