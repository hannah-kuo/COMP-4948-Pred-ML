from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
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
accuracy = []


def buildModel(batchSize):
    NUM_LAYERS = 7
    # define the keras model
    model = Sequential()
    model.add(Dense(230, input_dim=2, activation='relu', kernel_initializer='he_normal'))
    for i in range(0, NUM_LAYERS - 1):
        model.add(Dense(230, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(1, activation='sigmoid'))

    # opitimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9, name="SGD")
    opitimizer = tf.keras.optimizers.SGD(learning_rate=0.000001, momentum=0.7, name="SGD")

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer=opitimizer, metrics=['accuracy'])

    # fit the keras model on the dataset
    history = model.fit(X, y, epochs=500, batch_size=batchSize,
                        validation_data=(X_test, y_test))
    # evaluate the keras model

    # Evaluate the model.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: ' + str(acc) + ' Batch Size: ' + str(batchSize))
    accuracy.append(acc)
    print(accuracy)
    return history


def showLoss(history, batchSize):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history for training data.
    actualLabel = str(batchSize) + " batch"
    plt.subplot(1, 2, 1)
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actualLabel)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()


def showAccuracy(history, batchSize):
    # Get training and test loss histories
    training_loss = history.history['accuracy']
    validation_loss = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)

    actualLabel = str(batchSize) + " batch"
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actualLabel)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()


plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
batchSizes = [10, 32, 60, len(y_train)]
for i in range(0, len(batchSizes)):
    history = buildModel(batchSizes[i])
    showLoss(history, batchSizes[i])
    showAccuracy(history, batchSizes[i])

plt.show()
