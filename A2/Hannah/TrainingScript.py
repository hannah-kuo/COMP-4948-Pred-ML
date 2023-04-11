import matplotlib.pylab as plt
from keras import Model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import ElasticNet, LinearRegression
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import SMOTE
from keras.optimizers import Adam, SGD
from pathlib import Path
import pandas as pd
from sklearn import metrics
import numpy as np
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)

# --------------------------------------------------------------
# DATA LOADING & CLEANING
# --------------------------------------------------------------

# Load the dataset
PATH = "C:/PredML/A2/avocado.csv"
df = pd.read_csv(PATH)

# Preprocessing: Drop unnecessary columns and create dummy variables for categorical features
df = df.drop(columns=['Unnamed: 0', 'Date', 'type', 'region'])
df = pd.get_dummies(df, columns=['year'])

# Define the feature columns and target variable
feature_columns = ['4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
target_variable = 'AveragePrice'

# --------------------------------------------------------------
# Model 1: OLS with selected feature_columns + MinMaxScaler
# --------------------------------------------------------------

# Split data into train and test sets
X = df[feature_columns]
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled data back to DataFrames and assign original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

# Create a dictionary to store the results of each model
results = {}

# Make predictions and evaluate with the RMSE.
model = sm.OLS(y_train, X_train_scaled).fit()

predictions = model.predict(X_test_scaled)
# plot_loss_and_metrics("MinMaxScaled OLS Model (['age', 'annual Salary', 'net worth'])",y_test, predictions)
results['Model 1 OLS'] = {
    'R-squared': r2_score(y_test, predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
    'MSE': mean_squared_error(y_test, predictions),
    'MAE': mean_absolute_error(y_test, predictions)
}
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# --------------------------------------------------------------
# Model 2: Neural Network model
# --------------------------------------------------------------

# Neural Network model with 2 hidden layers
# nn_model = Sequential()
# nn_model.add(Dense(10, activation='relu', input_dim=len(X_train_scaled.columns)))  # input layer
# nn_model.add(Dense(10, activation='relu'))  # first hidden layer
# nn_model.add(Dense(10, activation='relu'))  # second hidden layer
# nn_model.add(Dense(1, activation='linear'))  # output layer
#
# nn_model.compile(optimizer='adam', loss='mean_squared_error')
# history = nn_model.fit(X_train_scaled, y_train, batch_size=16, epochs=50, validation_split=0.2)
# nn_predictions = nn_model.predict(X_test_scaled)
#
# results['Neural Network'] = {
#     'R-squared': r2_score(y_test, nn_predictions),
#     'RMSE': np.sqrt(mean_squared_error(y_test, nn_predictions)),
#     'MSE': mean_squared_error(y_test, nn_predictions),
#     'MAE': mean_absolute_error(y_test, nn_predictions)
# }
#
# # Print the results of each model
# for model_name, metrics in results.items():
#     print(model_name)
#     print(metrics)
#     print("\n")

# --------------------------------------------------------------
# Model 3: Neural Network model with Tuned Hyper-parameters
# --------------------------------------------------------------

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


# Function to create a neural network model
def create_model(layers, activation, kernel_initializer, learning_rate):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, activation=activation, input_dim=len(X_train_scaled.columns),
                            kernel_initializer=kernel_initializer))
        else:
            model.add(Dense(nodes, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dense(1, activation='linear', kernel_initializer=kernel_initializer))  # output layer
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


# Wrap the Keras model with KerasRegressor
# nn_model = KerasRegressor(model=create_model, verbose=0, learning_rate=0.001)
# nn_model = KerasRegressor(build_fn=create_model, verbose=0)
nn_model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the parameter search space
# param_space = {
#     'layers': [(10, 10), (20, 20), (30, 30), (10, 10, 10), (20, 20, 20), (30, 30, 30)],
#     'activation': ['relu', 'tanh'],
#     'kernel_initializer': ['glorot_uniform', 'he_uniform'],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'batch_size': [16, 32],
#     'epochs': [50, 100]
# }
param_space = {
    'layers': [(10, 10), (20, 20), (30, 30), (10, 10, 10), (20, 20, 20), (30, 30, 30)],
    'activation': ['relu', 'tanh'],
    'kernel_initializer': ['glorot_uniform', 'he_uniform'],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32],
    'epochs': [50, 100]
}

# Create the RandomizedSearchCV instance
random_search = RandomizedSearchCV(estimator=nn_model,
                                   param_distributions=param_space,
                                   n_iter=100,  # number of iterations
                                   cv=3,
                                   n_jobs=1,  # use all available cores
                                   verbose=2)

# Perform the random search
random_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters found
print("Best hyperparameters: ", random_search.best_params_)

# Train the model with the best hyperparameters and evaluate it
best_model = random_search.best_estimator_
nn_predictions = best_model.predict(X_test_scaled)

results['Neural Network'] = {
    'R-squared': r2_score(y_test, nn_predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, nn_predictions)),
    'MSE': mean_squared_error(y_test, nn_predictions),
    'MAE': mean_absolute_error(y_test, nn_predictions)
}

# Print the results of each model
for model_name, metrics in results.items():
    print(model_name)
    print(metrics)
    print("\n")

print("----------------------------------------")

""" 
# Prepare data for Model 2 and Model 3
X = df[feature_columns]
y = df[target_variable]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.transform(X_test)

from sklearn.model_selection import GridSearchCV

# Set the hyperparameters for GridSearchCV
param_grid = {
    'num_neurons': [10, 20, 30],
    'num_layers': [1, 2, 3],
    'activation_func': ['relu', 'tanh'],
    'kernel_initializer': ['uniform', 'normal']
}

from scikeras.wrappers import KerasRegressor


# Helper function to create the Neural Network model for GridSearchCV
def create_nn_model(num_neurons=10, num_layers=1, activation_func='relu', kernel_initializer='uniform'):
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=len(feature_columns), activation=activation_func,
                    kernel_initializer=kernel_initializer))
    for i in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation_func, kernel_initializer=kernel_initializer))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model


# Create the KerasRegressor and fit it to the training data
nn_model = KerasRegressor(
    build_fn=create_nn_model,
    epochs=100,
    batch_size=32,
    verbose=0,
    num_neurons=10,
    num_layers=1,
    activation_func='relu',
    kernel_initializer='uniform'
)

param_grid = {
    'epochs': [100],
    'batch_size': [32],
    'num_neurons': [10, 20],
    'num_layers': [1, 2],
    'activation_func': ['relu', 'tanh'],
    'kernel_initializer': ['uniform', 'normal']
}

grid_search = GridSearchCV(
    estimator=nn_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=0
)

# # verbose=2 helps us understand how long it might take to complete.
# grid_search = GridSearchCV(
#     estimator=nn_model,
#     param_grid=param_grid,
#     scoring='neg_mean_squared_error',
#     cv=3,
#     verbose=2
# )


grid_result = grid_search.fit(X_train_scaled, y_train)

# Remove these lines
# nn_model = KerasRegressor(build_fn=create_nn_model, epochs=100, batch_size=32, verbose=0)
# grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=0)
# grid_result = grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters from GridSearchCV
best_params = grid_result.best_params_
num_neurons = best_params['num_neurons']
num_layers = best_params['num_layers']
activation_func = best_params['activation_func']
kernel_initializer = best_params['kernel_initializer']

# Model 2
nn_model2 = Sequential()
nn_model2.add(Dense(10, activation='relu', input_dim=len(feature_columns)))
nn_model2.add(Dense(10, activation='relu'))
nn_model2.add(Dense(1, activation='linear'))

# Model 3
nn_model3 = Sequential()
nn_model3.add(Dense(num_neurons, input_dim=len(feature_columns), activation=activation_func,
                    kernel_initializer=kernel_initializer))
for i in range(num_layers - 1):
    nn_model3.add(Dense(num_neurons, activation=activation_func, kernel_initializer=kernel_initializer))
nn_model3.add(Dense(1, activation='linear'))
"""
