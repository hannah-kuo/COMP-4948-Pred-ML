import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
#
# PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/housing.data")
# df = pd.read_csv(PATH, header=None)
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
#
# # Increase number of columns that display on one line.
# pd.set_option('display.width', 1000)
#
# print(df.head())
# print(df.tail())
# print(df.describe())
#
# dataset = df.values
#
# # split into input (X) and output (Y) variables
# X = dataset[:, 0:13]
# y = dataset[:, 13]
#
# ROW_DIM = 0
# COL_DIM = 1
#
# x_arrayReshaped = X.reshape(X.shape[ROW_DIM],
#                             X.shape[COL_DIM])
#
# # Convert DataFrame columns to vertical columns of target variables values.
# y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)
#
# X_train, X_temp, y_train, y_temp = train_test_split(x_arrayReshaped,
#                                                     y_arrayReshaped, test_size=0.3, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_temp,
#                                                 y_temp, test_size=0.5, random_state=0)
#
# n_features = X_train.shape[1]
#
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

#
#
# # Define the model.
# # def create_model(numNeurons=3, initializer='normal', activation='relu'):
# #     # create model
# #     model = Sequential()
# #     model.add(Dense(25, kernel_initializer='normal',
# #                     input_dim=n_features, activation='relu'))
# #
# #     model.add(Dense(numNeurons, kernel_initializer=initializer,
# #                     activation=activation))
# #
# #     model.add(Dense(1, kernel_initializer='normal', activation='relu'))
# #     opt = tensorflow.keras.optimizers.Adam(learning_rate=0.005)
# #     # Compile model
# #     model.compile(loss='mse', optimizer=opt)
# #     return model
#
# def create_model(numNeurons=50, numHiddenLayers=1, initializer='normal', activation='relu'):
#     # create model
#     model = Sequential()
#     model.add(Dense(25, kernel_initializer='uniform',
#                     input_dim=n_features, activation='softplus'))
#
#     for i in range(numHiddenLayers):
#         model.add(Dense(numNeurons, kernel_initializer=initializer,
#                         activation=activation))
#
#     model.add(Dense(1, kernel_initializer='he_normal',  activation='softplus'))
#     opt = tensorflow.keras.optimizers.Adam(learning_rate=0.005)
#     # Compile model
#     model.compile(loss='mse', optimizer=opt)
#     return model
#
# ### Grid Building Section #######################
# # Define the parameters to try out
# # params = {'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh',
# #                          'sigmoid', 'hard_sigmoid', 'linear'],
# #           'numNeurons': [10, 15, 20, 25, 30, 35],
# #           'initializer': ['uniform', 'lecun_uniform', 'normal', 'zero',
# #                           'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# #           }
#
# params = { 'activation' : ['relu', 'sigmoid'],
#            'numNeurons':   [50, 100, 200],
#            'numHiddenLayers':[1,2,3],
#            'initializer': ['normal']
#           }
#
#
# model = KerasRegressor(build_fn=create_model, epochs=10,
#                        batch_size=9, verbose=1)
# # grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3)
#
# from sklearn.model_selection     import RandomizedSearchCV
# grid = RandomizedSearchCV(model, param_distributions = params, cv = 3)
#
# #################################################
#
# grid_result = grid.fit(X_train, y_train)
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
#
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
# from sklearn                 import metrics
# predictions = grid.best_estimator_.predict(X_test)
# print(predictions)
# mse         = metrics.mean_squared_error(y_test, predictions)
# print("Neural network MSE: " + str(mse))
# print("Neural network RMSE: " + str(np.sqrt(mse)))


"""Example 3: Visualizing Loss and Accuracy"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from tensorflow import keras
# from tensorflow.keras.layers import Dense

# PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/iris_old.csv")
# df         = pd.read_csv(PATH)
# df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']
# print(df)
#
# # Convert text to numeric category.
# # 0 is setosa, 1 is versacolor and 2 is virginica
# df['y'] = LabelEncoder().fit_transform(df['Iris Type'])
#
# # Prepare the data.
# dfX = df.iloc[:, 0:4] # Get X features only from columns 0 to 3
# dfY = df.iloc[:, 5:6] # Get X features only from column 5
#
# ROW_DIM = 0
# COL_DIM = 1
#
# # Create vertical array of features.
# x_array = dfX.values
# x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
#                                   x_array.shape[COL_DIM])
#
# y_array = dfY.values
# y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM],
#                                   y_array.shape[COL_DIM])


# import pandas                as     pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing   import LabelEncoder
# from tensorflow.keras        import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
from pathlib import Path

# PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/bill_authentication.csv")
# df = pd.read_csv(PATH)
#
# # Convert text to numeric category.
# # 0 is setosa, 1 is versacolor and 2 is virginica
# y = df['Class']
# X = df
# del X['Class']
# ROW_DIM = 0
# COL_DIM = 1
#
# # Create vertical array of features.
# x_array = X.values
# x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
#                                   x_array.shape[COL_DIM])
#
# y_array = np.array(y.values)
# y_arrayReshaped = y_array.reshape(len(y_array), 1)
#
# # Split into train, validation and test data sets.
# X_train, X_temp, y_train, y_temp = train_test_split(
#     x_arrayReshaped, y_arrayReshaped, test_size=0.33)
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.50)
#
# n_features = X_train.shape[COL_DIM]
#
# # Define the model.
# model = Sequential()
#
# # Hidden layer 1 (also receives the input layer)
# model.add(Dense(2, activation='relu', input_shape=(n_features,)))
#
# # Output layer
# model.add(Dense(3, activation='softmax'))
#
# # Compile the model.
# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Fit the model.
# history = model.fit(X_train, y_train, epochs=1000, batch_size=28, verbose=1,
#                     validation_data=(X_val, y_val))
#
# # Evaluate the model with unseen data.
# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print('Test Accuracy: %.3f' % acc)
#
# # make a prediction
# row = [5.1, 3.5, 1.4, 0.2]
# yhat = model.predict([row])
# print('Predicted: s (class=d)' + str(yhat))
#
# import matplotlib.pyplot as plt
#
#
# def showLoss(history):
#     # Get training and test loss histories
#     training_loss = history.history['loss']
#     validation_loss = history.history['val_loss']
#
#     # Create count of the number of epochs
#     epoch_count = range(1, len(training_loss) + 1)
#     plt.subplot(1, 2, 1)
#     # Visualize loss history for training data.
#     plt.plot(epoch_count, training_loss, label='Train Loss', color='red')
#
#     # View loss on unseen data.
#     plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
#              color='black')
#
#     plt.xlabel('Epoch')
#     plt.legend(loc="best")
#     plt.title("Loss")
#
#
# def showAccuracy(history):
#     # Get training and test loss histories
#     training_loss = history.history['accuracy']
#     validation_loss = history.history['val_accuracy']
#
#     # Create count of the number of epochs
#     epoch_count = range(1, len(training_loss) + 1)
#     plt.subplot(1, 2, 2)
#     # Visualize loss history for training data.
#     plt.plot(epoch_count, training_loss, label='Train Accuracy', color='red')
#
#     # View loss on unseen data.
#     plt.plot(epoch_count, validation_loss, 'r--',
#              label='Validation Accuracy', color='black')
#     plt.xlabel('Epoch')
#     plt.legend(loc="best")
#     plt.title('Accuracy')
#
#
# plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
# showLoss(history)
# showAccuracy(history)
# plt.show()
#
# from sklearn.metrics import classification_report
#
# # Provide detailed evaluation with unseen data.
# y_probability = model.predict(X_test)
# import numpy as np
#
# # Convert probability arrays to whole numbers.
# # eg. [0.0003, 0.01, 0.9807] becomes 2.
# predictions = np.argmax(y_probability, axis=-1)
# print(classification_report(y_test, predictions))


"""EXAMPLE 4  Exercise 6 """

# import pandas as pd
# import numpy  as np
# from sklearn                 import metrics
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
# PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/housing.data")
#
# df = pd.read_csv(PATH, header=None)
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
#
# # Increase number of columns that display on one line.
# pd.set_option('display.width', 1000)
#
# print(df.head())
# print(df.tail())
# print(df.describe())
#
# dataset = df.values
# # split into input (X) and output (Y) variables
# X       = dataset[:, 0:13]
# y       = dataset[:, 13]
#
# ROW_DIM = 0
# COL_DIM = 1
# x_arrayReshaped = X.reshape(X.shape[ROW_DIM],
#                             X.shape[COL_DIM])
#
# # Convert DataFrame columns to vertical columns of target variables values.
# y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)
#
# # Splitting data into 3 data sets.
# # Train and val are used while model is being fit.
# # Test is held back until the end for the final evaluation.
# X_train, X_temp, y_train, y_temp = train_test_split(x_arrayReshaped,
#                      y_arrayReshaped, test_size=0.3, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_temp,
#                      y_temp, test_size=0.3, random_state=0)
#
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
#
# # define base model
# from keras.optimizers import Adam   #for adam optimizer
# def baseline_model():
#     model = Sequential()
#     model.add(Dense(25, input_dim=13, kernel_initializer='uniform',
#                         activation='softplus'))
#     model.add(Dense(10, kernel_initializer='lecun_uniform', activation='softplus'))
#     model.add(Dense(1,  kernel_initializer='uniform'))
#
#     # Use Adam optimizer with the given learning rate
#     opt = Adam(lr=0.005)
#     model.compile(loss='mean_squared_error')
#     return model
#
# # evaluate model
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100,
#                            batch_size=9, verbose=1)
# kfold   = KFold(n_splits=10)
# results = cross_val_score(estimator, X_train, y_train, cv=kfold)
# print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
# print("Baseline RMSE: " + str(np.sqrt(results.std())))
#
# # So then we build the model.
# model = baseline_model()
# history = model.fit(X_train, y_train, epochs=100,
#                     batch_size=9, verbose=1,
#                     validation_data=(X_val, y_val))
# predictions = model.predict(X_test)
#
# mse = metrics.mean_squared_error(y_test, predictions)
# print("Neural network MSE: " + str(mse))
# print("Neural network RMSE: " + str(np.sqrt(mse)))
# def showLoss(history):
#     # Get training and test loss histories
#     training_loss = history.history['loss']
#     validation_loss = history.history['val_loss']
#
#     # Create count of the number of epochs
#     epoch_count = range(1, len(training_loss) + 1)
#     plt.subplot(1, 2, 1)
#     # Visualize loss history for training data.
#     plt.plot(epoch_count, training_loss, label='Train Loss', color='red')
#
#     # View loss on unseen data.
#     plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
#              color='black')
#
#     plt.xlabel('Epoch')
#     plt.legend(loc="best")
#     plt.title("Loss")
#
#
# def showAccuracy(history):
#     # Get training and test loss histories
#     training_loss = history.history['accuracy']
#     validation_loss = history.history['val_accuracy']
#
#     # Create count of the number of epochs
#     epoch_count = range(1, len(training_loss) + 1)
#     plt.subplot(1, 2, 2)
#     # Visualize loss history for training data.
#     plt.plot(epoch_count, training_loss, label='Train Accuracy', color='red')
#
#     # View loss on unseen data.
#     plt.plot(epoch_count, validation_loss, 'r--',
#              label='Validation Accuracy', color='black')
#     plt.xlabel('Epoch')
#     plt.legend(loc="best")
#     plt.title('Accuracy')
#
#
# plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
# showLoss(history)
# # showAccuracy(history)
# plt.show()


"""Example 5,6   Exercise 7"""

# from sklearn import metrics
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# plt.style.use('ggplot')
#
#
# import matplotlib.pyplot as plt
#
# PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/iris_v2.csv")
# # Create numeric target for iris type.
# dataset = pd.read_csv(PATH)
# dataset.iris_type = pd.Categorical(dataset.iris_type)
#
# # Prepare x and y.
# dataset['flowertype'] = dataset.iris_type.cat.codes
# del dataset['iris_type']
# y = dataset['flowertype']
# X = dataset
# del X['flowertype']
#
# # Split X and y.
# X_train, X_test, y_train, y_test = train_test_split(X,
#                                     y, test_size=0.30)
# # Scale X and Y.
# scX=StandardScaler()
# scalerX = scX.fit(X_train)
# trainX_scaled = scalerX.transform(X_train)
# testX_scaled  = scalerX.transform(X_test)
#
# # Create and fit model.
# model = MLPClassifier()
# model.fit(trainX_scaled, y_train)
# print(model.get_params()) # Show model parameters.
#
#
# # Evaluate model.
# predicted_y = model.predict(testX_scaled)
# print(metrics.classification_report(y_test, predicted_y))
# print(metrics.confusion_matrix(y_test, predicted_y))
#
# def showLosses(model):
#     plt.plot(model.loss_curve_)
#     plt.title("Loss Curve")
#     plt.xlabel('Iterations')
#     plt.ylabel('Cost')
#     plt.show()
# showLosses(model)
#
# parameters={
# 'solver': ['adam', 'sgd'],
# 'learning_rate': ['constant', 'adaptive', 'invscaling'],
# 'hidden_layer_sizes': [(200,200), (300,200), (150,150)],
# 'activation': ["logistic", "relu", "tanh"]
# }
# model2 = GridSearchCV(estimator=model, param_grid=parameters,
#                             scoring='accuracy', # average='macro'),
#                             n_jobs=-1, cv=4, verbose=1,
#                             return_train_score=False)
#
# model2.fit(trainX_scaled, y_train)
# print("Best parameters: ")
# print(model2.best_params_)
# y_pred = model2.predict(testX_scaled)
#
# print("Report with grid: ")
# print(metrics.classification_report(y_test, y_pred))
# print(metrics.confusion_matrix(y_test, y_pred))
# showLosses(model2.best_estimator_)


"""Example 7: MLP Regressor"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import warnings

warnings.filterwarnings(action='once')

PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/housing.data")

df = pd.read_csv(PATH,header=None)

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
# param_grid = {
#     'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
#     'max_iter':            [50, 100],
#     'activation':          ['tanh', 'relu'],
#     'solver':              ['sgd', 'adam'],
#     'alpha':               [0.0001, 0.05],
#     'learning_rate':       ['constant','adaptive'],
# }


# here is the new part.
param_grid = {
    'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
    'max_iter': [50, 100],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2]
}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# n_jobs=-1 means use all processors.
# Run print(metrics.get_scorer_names()) for scoring choices.
model2 = MLPRegressor()
# gridModel = GridSearchCV(model2, param_grid, n_jobs=-1, cv=10,scoring='neg_mean_squared_error')

gridModel = RandomizedSearchCV(model2, param_distributions=param_grid, n_jobs=-1, scoring='neg_mean_squared_error')

gridModel.fit(trainX_scaled, trainY_scaled)

print("Best parameters")
print(gridModel.best_params_)
evaluateModel(gridModel.best_estimator_, valX_scaled, valY_scaled, scY)

# Evaluate both models with test (unseen) data.
print("\n*** Base model with test data: ")
evaluateModel(model1, testX_scaled, testY_scaled, scY)
print(model1.get_params())
print("\n*** Grid searched model with test data: ")
evaluateModel(gridModel.best_estimator_, testX_scaled, testY_scaled, scY)
print(gridModel.get_params())
