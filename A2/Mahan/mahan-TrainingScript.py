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

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)




### CODE USED TO SPLIT THE ORIGINAL CSV FILE INTO TEST AND TRAIN SETS#####

def split_csv_to_train_test(input_file, train_percentage=0.8, test_percentage=0.2):
    # Read input CSV file
    data = pd.read_csv(input_file, encoding="ISO-8859-1", sep=',')

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, train_size=train_percentage, test_size=test_percentage, random_state=42)

    # Save train and test data to CSV files
    train_data.to_csv('C:/PredML/A2/mahan/train.csv', index=False)
    test_data.to_csv('C:/PredML/A2/mahan/test.csv', index=False)

# PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/car_purchasing.csv")
# split_csv_to_train_test(PATH, train_percentage=0.8, test_percentage=0.2)




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

################################# DATA LOADING & CLEANING #################################
# PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/car_purchasing.csv")
PATH = "C:/PredML/A2/Car_Purchasing_Data.csv"
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df = pd.read_csv("C:/PredML/A2/mahan/train.csv", encoding="ISO-8859-1", sep=',')
df = df.drop_duplicates()
df.drop(columns=['customer name', 'customer e-mail', 'country'], inplace=True)
df = pd.get_dummies(df, columns=['gender'])
# print("Null\n", df.isnull().sum())
# print(df.info())
# print(df.head())


def remove_outlier(df, col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    df_out = df.loc[(df[col_name] > Q1 - 1.5 * IQR) & (df[col_name] < Q3 + 1.5 * IQR)]
    return df_out

df = remove_outlier(df, 'age')
df = remove_outlier(df, 'credit card debt')
df = remove_outlier(df, 'net worth')
df = remove_outlier(df, 'annual Salary')

results = {}



#
# ################################## MODELS #################################
#



#### Model 1: OLS with ['age', 'annual Salary', 'net worth'] + MinMaxScaler

# Split data into train and test sets
X = df[['age', 'annual Salary', 'net worth']]
y = df['car purchase amount']

X = sm.add_constant(X)  # double check this is needed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Make predictions and evaluate with the RMSE.
model = sm.OLS(y_train, X_train_scaled).fit()

# OLS_model = model
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



# Split data into train and test sets (Have to redo in order to remove the constant added for OLS above)
X = df[['age', 'annual Salary', 'net worth']]
y = df['car purchase amount']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Split into train, validation and test data sets.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.2)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.transform(X_test)

#### Model 2: NN 3 layers: 2 hidden layers with 10 nodes each and a ReLU activation function, and one output layer with a single node and a linear activation function. The model is compiled with an optimizer set to 'adam' and a loss function set to 'mean_squared_error'.



model = Sequential()
model.add(Dense(10, activation='relu', input_dim=3))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error')  # ,metrics=['mean_absolute_error'])
history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_split=0.2, validation_data=(X_val, y_val))

model.save('BinaryFolder/basic_sequential_NN_model.pkl')

y_pred = model.predict(X_test)

basic_sequential_NN_model = model

results['Model 2 NN (3layers)'] = {
    'R-squared': r2_score(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MSE': mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred)
}




plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training or Validation (M2 NN)')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

loss = model.evaluate(X_test, y_test)
print("Multi-Layer Perceptron (MLP) loss: ", loss)


#### MODEL 3 #################################################

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                y_temp, test_size=0.5, random_state=0)

# set best parameters
num_layers = 2
num_neurons = 16
activation_func = 'relu'
learning_rate = 0.01
kernel_initializer = 'uniform'

# create model with best parameters
model = Sequential()
model.add(
    Dense(num_neurons, input_dim=X_train.shape[1], activation=activation_func, kernel_initializer=kernel_initializer))
for i in range(num_layers - 1):
    model.add(Dense(num_neurons, activation=activation_func, kernel_initializer=kernel_initializer))
model.add(Dense(1, activation='linear'))
opt = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=opt, metrics=['mae'])
from sklearn.metrics import mean_squared_error

# Fit the model.
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# simple early stopping
# patience:  # of epochs observed where no improvement before exiting.
# mode:      Could be max, min, or auto.
# min_delta: Amount of change needed to be considered an improvement.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.000001, patience=200)
mc = ModelCheckpoint('BinaryFolder/best_model2.h5', monitor='val_loss', mode='min', verbose=1,
                     save_best_only=True)

# fit model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=4000, verbose=0,
                    callbacks=[es, mc])
# history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_split=0.2, validation_data=(X_val, y_val))
model.save('BinaryFolder/sequential_NN_model.pkl')
# load the saved model
model = load_model('BinaryFolder/best_model2.h5')

sequential_NN_model = model

predictions = model.predict(X_test)

results['Model 3 gridchosen param'] = {
    'R-squared': r2_score(y_test, predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
    'MSE': mean_squared_error(y_test, predictions),
    'MAE': mean_absolute_error(y_test, predictions)
}

print("RMSE m3: " + str(np.sqrt(mean_squared_error(y_test, predictions))))  # evaluate model on validation set
val_loss = model.evaluate(X_val, y_val)[0]
print('Validation loss M3:', val_loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training or Validation (gridchosen param-M3)')
plt.ylabel('Training & Validation Losses')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()



##########MODEL 4 MLPRegresson#######################


# Create and fit model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = MLPRegressor()
mplpregressor_model = model
model.fit(X_train, y_train)
print("=======MLPRegressor======")
print(model.get_params())  # Show model parameters.

# Evaluate model.
predicted_y = model.predict(X_test)
results['Model 4 MLPRegressor'] = {
    'R-squared': r2_score(y_test, predicted_y),
    'RMSE': np.sqrt(mean_squared_error(y_test, predicted_y)),
    'MSE': mean_squared_error(y_test, predicted_y),
    'MAE': mean_absolute_error(y_test, predicted_y)
}

print("RMSE m3: " + str(np.sqrt(mean_squared_error(y_test, predicted_y))))  # evaluate model on validation set
showLosses(model)

print("=======MLPRegressor ENDS======")


######## MODEL 5: FINAL STACKED MODEL, USES SOME OF THE ABOVE MODELS ###############################

print("=======START OF STACKED MODEL CODE======")


def getUnfitModels():
    models = list()
    models.append(ElasticNet())
    models.append(DecisionTreeRegressor())
    models.append(AdaBoostRegressor())
    models.append(RandomForestRegressor(n_estimators=200))
    models.append(ExtraTreesRegressor(n_estimators=200))
    models.append(mplpregressor_model)
    models.append(basic_sequential_NN_model)
    models.append(sequential_NN_model)
    return models


def evaluateModel(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse), 3)
    rsquared = r2_score(y_test, predictions)
    print(" RMSE:" + str(rmse) + " R2:  " + str(rsquared) + " " + model.__class__.__name__)


def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

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

# Save stacked model
# joblib.dump(stackedModel, "stacked_model.pkl")



"""GRID SEARCH FOR BEST PARAM FOR M3 ANN"""

"""Initial result for me: Best Param: {'num_layers': 2, 'num_neurons': 16, 'activation_func': 'relu', 'learning_rate': 0.01, 'kernel_initializer': 'uniform', 'train_loss': 45131624.0, 'val_loss': 33958648.0}

"""


def grid_search_nn(X_train, y_train, X_val, y_val, hidden_layers, neurons, activations, learning_rates, kernel_initializers):
    best_loss = float('inf')
    best_params = None
    results = []
    for layer_size in hidden_layers:
        for num_neurons in neurons:
            for activation_func in activations:
                for learning_rate in learning_rates:
                    for kernel_initializer in kernel_initializers:
                        model = Sequential()
                        model.add(Dense(num_neurons, input_dim=X_train.shape[1], activation=activation_func, kernel_initializer=kernel_initializer))
                        for i in range(layer_size - 1):
                            model.add(Dense(num_neurons, activation=activation_func, kernel_initializer=kernel_initializer))
                        model.add(Dense(1, activation='linear'))
                        opt = Adam(learning_rate=learning_rate)
                        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
                        history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=0)
                        train_loss = history.history['loss'][-1]
                        val_loss = history.history['val_loss'][-1]
                        results.append({
                            'num_layers': layer_size,
                            'num_neurons': num_neurons,
                            'activation_func': activation_func,
                            'learning_rate': learning_rate,
                            'kernel_initializer': kernel_initializer,
                            'train_loss': train_loss,
                            'val_loss': val_loss
                        })
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_params = {
                                'num_layers': layer_size,
                                'num_neurons': num_neurons,
                                'activation_func': activation_func,
                                'learning_rate': learning_rate,
                                'kernel_initializer': kernel_initializer,
                                'train_loss': train_loss,
                                'val_loss': val_loss
                            }
    return results, best_params


""" RUN THE BELOW CODE TO CALL grid_search_nn()"""

#
#
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# # Load your dataset and split it into training, validation, and test sets
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
#
# # Define the hyperparameters to search over
# hidden_layers = [1, 2, 3]
# neurons = [8, 16, 32]
# activations = ['relu', 'sigmoid']
# learning_rates = [0.001, 0.01, 0.1]
# kernel_initializers = ['uniform', 'normal', 'glorot_uniform']
# import datetime
#
# # Record the start time
# start_time = datetime.datetime.now()
# # Call the grid_search_nn function
# results, bestParam = grid_search_nn(X_train, y_train, X_val, y_val, hidden_layers, neurons, activations, learning_rates, kernel_initializers)
#
# # Print the results
# for r in results:
#     print(r)
# print("Best Param:", bestParam)
# # Record the end time
# end_time = datetime.datetime.now()
#
# # Calculate the elapsed time
# elapsed_time = end_time - start_time
#
# # Print the elapsed time
# print("Elapsed time: ", elapsed_time)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)\
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
