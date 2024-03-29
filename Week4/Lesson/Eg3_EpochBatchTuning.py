""" Example 3: Epoch and Batch Tuning """

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

PATH = "C:/PredML/"
CSV_DATA = "housing.data"
df = pd.read_csv(PATH + CSV_DATA, header=None)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
y = dataset[:, 13]

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                y_temp, test_size=0.5, random_state=0)


def evaluateModel(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("RMSE: " + str(rmse))
    return rmse


def showResults(networkStats):
    dfStats = pd.DataFrame.from_records(networkStats)
    dfStats = dfStats.sort_values(by=['rmse'])
    print(dfStats)


networkStats = []

# ------------------ Model parameters ----------------------------
batch_sizes = [10, 60, 100]
epochList = [50, 100, 200]


# --------------------------------------------------------------

# --------------------------------------------------------------
# Build model
def create_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


for batch_size in batch_sizes:
    for epochs in epochList:
        model = create_model()
        history = model.fit(X_train, y_train, epochs=epochs,
                            batch_size=batch_size, verbose=1,
                            validation_data=(X_val, y_val))
        rmse = evaluateModel(model, X_test, y_test)
        networkStats.append({"rmse": rmse, "epochs": epochs, "batch": batch_size})
showResults(networkStats)
# --------------------------------------------------------------
