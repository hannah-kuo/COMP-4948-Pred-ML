import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf

PATH = "C:/PredML/fluDiagnosis.csv"

# load the dataset
df = pd.read_csv(PATH)
# split into input (X) and output (y) variables
print(df)

X = df[['A', 'B']]
y = df[['Diagnosed']]
# Split into train and test data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


def buildModel(numLayers):
    # define the keras model
    model = Sequential()
    model.add(Dense(230, input_dim=2, activation='relu',
                    kernel_initializer='he_normal'))

    for i in range(0, numLayers - 1):
        model.add(Dense(230, activation='relu',
                        kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    opitimizer = tf.keras.optimizers.SGD(
        learning_rate=0.00001, momentum=0.8, name="SGD",
    )

    # Compile the keras model.
    model.compile(loss='binary_crossentropy', optimizer=opitimizer,
                  metrics=['accuracy'])

    # Fit the keras model on the dataset.
    history = model.fit(X_train, y_train, epochs=200, batch_size=10,
                        validation_data=(X_test, y_test))

    # Evaluate the model.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)
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
    plt.legend()


numLayers = [5, 6, 7, 8]
plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

for i in range(0, len(numLayers)):
    history = buildModel(numLayers[i])
    showLoss(history, numLayers[i])
    showAccuracy(history, numLayers[i])

plt.show()
