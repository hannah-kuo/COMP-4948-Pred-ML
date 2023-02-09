""" Example 7: MLP Regressor"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import warnings

warnings.filterwarnings(action='once')

PATH = "C:/PredML/"
CSV_DATA = "housing.data"
df = pd.read_csv(PATH + CSV_DATA, header=None)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
y = dataset[:, 13]

trainX, temp_X, trainY, temp_y = train_test_split(X, y, train_size=0.7)
valX, testX, valY, testY = train_test_split(temp_X, temp_y, train_size=0.5)

# Scale X and Y.
scX = StandardScaler()
scalerX = scX.fit(trainX)
trainX_scaled = scalerX.transform(trainX)
valX_scaled = scalerX.transform(valX)
testX_scaled = scalerX.transform(testX)

scY = StandardScaler()
trainY_scaled = scY.fit_transform(np.array(trainY).reshape(-1, 1))
testY_scaled = scY.transform(np.array(testY).reshape(-1, 1))
valY_scaled = scY.transform(np.array(valY).reshape(-1, 1))

# Build basic multilayer perceptron.
model1 = MLPRegressor(
    # 3 hidden layers with 150 neurons, 100, and 50.
    hidden_layer_sizes=(150, 100, 50),
    max_iter=50,  # epochs
    activation='relu',
    solver='adam',  # optimizer
    verbose=1)
model1.fit(trainX_scaled, trainY_scaled)


def showLosses(model):
    plt.plot(model.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


def evaluateModel(model, testX_scaled, testY_scaled, scY):
    showLosses(model)
    scaledPredictions = model.predict(testX_scaled)
    y_pred = scY.inverse_transform(
        np.array(scaledPredictions).reshape(-1, 1))
    mse = metrics.mean_squared_error(testY_scaled, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))


evaluateModel(model1, valX_scaled, valY_scaled, scY)

# here is the new part.
param_grid = {
    'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
    'max_iter': [50, 100],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}
