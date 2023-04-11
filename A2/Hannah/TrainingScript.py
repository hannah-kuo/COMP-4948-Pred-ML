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
from keras.optimizers import Adam, RMSprop
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
nn_model = Sequential()
nn_model.add(Dense(10, activation='relu', input_dim=len(X_train_scaled.columns)))  # input layer
nn_model.add(Dense(10, activation='relu'))  # first hidden layer
nn_model.add(Dense(10, activation='relu'))  # second hidden layer
nn_model.add(Dense(1, activation='linear'))  # output layer

nn_model.compile(optimizer='adam', loss='mean_squared_error')
history = nn_model.fit(X_train_scaled, y_train, batch_size=16, epochs=50, validation_split=0.2)
nn_predictions = nn_model.predict(X_test_scaled)

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


# --------------------------------------------------------------
# Grid Search for Optimal Neural Network Hyper-parameters
# --------------------------------------------------------------

# Create NN Model
# def create_nn_model(optimizer='adam', neurons=10, lr=0.001, activation='relu', initializer='he_normal'):
#     model = Sequential()
#     model.add(Dense(neurons, activation=activation, kernel_initializer=initializer, input_dim=8))
#     model.add(Dense(neurons, activation=activation, kernel_initializer=initializer))
#     model.add(Dense(1, activation='linear'))
#
#     if optimizer == 'adam':
#         opt = Adam(learning_rate=lr)
#     elif optimizer == 'rmsprop':
#         opt = RMSprop(learning_rate=lr)
#
#     model.compile(optimizer=opt, loss='mean_squared_error')
#     return model
#
#
# # Define the grid search parameters
# param_grid = {
#     'optimizer': ['adam', 'rmsprop'],
#     'neurons': [10, 20, 30],
#     'lr': [0.001, 0.01, 0.1],
#     'activation': ['relu', 'tanh'],
#     'initializer': ['he_normal', 'he_uniform']
# }
#
# # Perform Grid Search
# # Create a KerasRegressor model
# nn_model = KerasRegressor(build_fn=create_nn_model, epochs=50, batch_size=16, verbose=0, optimizer='adam', neurons=10,
#                           lr=0.001, activation='relu', initializer='he_normal')
#
# # Define the search space
# search_space = {
#     'optimizer': Categorical(['adam', 'rmsprop']),
#     'neurons': Integer(10, 100),
#     'activation': Categorical(['relu', 'tanh']),
#     'lr': Real(1e-4, 1e-2, prior='log-uniform'),
# }
#
# # Perform the Grid search
# grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
# grid_search_result = grid_search.fit(X_train_scaled, y_train)
#
# # Summarize the results
# print(f"Best: {grid_search_result.best_score_} using {grid_search_result.best_params_}")
# means = grid_search_result.cv_results_['mean_test_score']
# stds = grid_search_result.cv_results_['std_test_score']
# params = grid_search_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print(f"{mean:.4f} ({stdev:.4f}) with: {param}")

# --------------------------------------------------------------
# Model 3: Neural Network model with Tuned Hyper-parameters
# --------------------------------------------------------------


# Define the model with best parameters
def create_model(layers, activation, kernel_initializer, learning_rate):
    model = Sequential()
    for i, layer_size in enumerate(layers):
        if i == 0:
            model.add(Dense(layer_size, input_shape=(X_train_scaled.shape[1],), kernel_initializer=kernel_initializer,
                            activation=activation))
        else:
            model.add(Dense(layer_size, kernel_initializer=kernel_initializer, activation=activation))
    model.add(Dense(1, kernel_initializer=kernel_initializer, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


# Best hyperparameters from the grid search
layers = (30, 30)  # 2 hidden layers
activation = 'relu'
kernel_initializer = 'he_uniform'
learning_rate = 0.001
batch_size = 32  # You can adjust this value if you want
epochs = 100  # You can adjust this value if you want

# Create the third model
third_model = create_model(layers, activation, kernel_initializer, learning_rate)

# Train the third model
third_model_history = third_model.fit(X_train_scaled, y_train, batch_size=batch_size, epochs=epochs,
                                      validation_split=0.2)

# Make predictions with the third model
third_model_predictions = third_model.predict(X_test_scaled)

# Evaluate the third model
results['Third Model'] = {
    'R-squared': r2_score(y_test, third_model_predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, third_model_predictions)),
    'MSE': mean_squared_error(y_test, third_model_predictions),
    'MAE': mean_absolute_error(y_test, third_model_predictions)
}

# Print the results of each model
for model_name, metrics in results.items():
    print(model_name)
    print(metrics)
    print("\n")

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001, patience=200)
mc = ModelCheckpoint('BinaryFolder/best_model3.h5', monitor='val_loss', mode='min', verbose=1,
                     save_best_only=True)
third_model_history = third_model.fit(X_train_scaled, y_train, batch_size=batch_size, epochs=epochs,
                                      validation_split=0.2, callbacks=[es, mc])
third_model.save('BinaryFolder/sequential_NN_model3.pkl')
from keras.models import load_model

best_third_model = load_model('BinaryFolder/best_model3.h5')
third_model_predictions = best_third_model.predict(X_test_scaled)

results['Model 3 grid-chosen param; Loaded from pkl'] = {
    'R-squared': r2_score(y_test, third_model_predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, third_model_predictions)),
    'MSE': mean_squared_error(y_test, third_model_predictions),
    'MAE': mean_absolute_error(y_test, third_model_predictions)
}

print("RMSE m3: " + str(np.sqrt(mean_squared_error(y_test, third_model_predictions))))

val_loss = third_model.evaluate(X_test_scaled, y_test)
print('Validation loss M3:', val_loss)

plt.plot(third_model_history.history['loss'])
plt.plot(third_model_history.history['val_loss'])
plt.title('Model Loss During Training or Validation (gridchosen param-M3)')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# --------------------------------------------------------------
# Model 4: Stacked Model
# --------------------------------------------------------------

print("=======START OF STACKED MODEL CODE======")


# TODO: Change the combo here
def getUnfitModels():
    models = list()
    models.append(ElasticNet())
    models.append(DecisionTreeRegressor())
    models.append(AdaBoostRegressor())
    models.append(RandomForestRegressor(n_estimators=200))
    models.append(ExtraTreesRegressor(n_estimators=200))
    models.append(nn_model)
    models.append(third_model)
    return models


def evaluateModel(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse), 3)
    rsquared = r2_score(y_test, predictions)
    print(" RMSE:" + str(rmse) + " R2:  " + str(rsquared) + " " + model.__class__.__name__)


def fitBaseModels(X_train, y_train, X_val, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        if isinstance(models[i], Sequential):
            models[i].fit(X_train_scaled, y_train)
            predictions = models[i].predict(X_val_scaled)
        else:
            models[i].fit(X_train, y_train)
            predictions = models[i].predict(X_val)

        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions[:, 0] if isinstance(models[i], Sequential) else predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def plot_loss_and_metrics(model_name, y_true, y_pred):
    print(f"====Model {model_name} ====")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f'R^2 Score: {r2:.4f}')
    print(f"==== # ====")


def showLosses(model):
    plt.plot(model.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
stackedModel = fitStackedModel(dfPredictions, y_val)
print("Object type before pickling:", type(stackedModel))
joblib.dump(stackedModel, "BinaryFolder/stacked_model.pkl")

# Evaluate base models with validation data.
print("\n** Evaluate Base Models **")
dfValidationPredictions = pd.DataFrame()
for i in range(0, len(models)):
    predictions = models[i].predict(X_test)
    colName = str(i)
    dfValidationPredictions[colName] = predictions
    evaluateModel(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
stackedPredictions = stackedModel.predict(dfValidationPredictions)
print("\n** Evaluate Stacked Model **")
evaluateModel(y_test, stackedPredictions, stackedModel)
showLosses(model)

results['Model 5 Stacked Linear Regression'] = {
    'R-squared': r2_score(y_test, stackedPredictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, stackedPredictions)),
    'MSE': mean_squared_error(y_test, stackedPredictions),
    'MAE': mean_absolute_error(y_test, stackedPredictions)
}

# Print results
for model_temp, metrics in results.items():
    print(model_temp)
    print(metrics)
    print("\n")

# Save base models and scalers
for i, model in enumerate(unfitModels):
    if i == 6:
        break
    else:
        joblib.dump(model, f"BinaryFolder/base_model_{i}.pkl")

# Save MinMaxScaler
joblib.dump(scaler, "BinaryFolder/scaler.pkl")
