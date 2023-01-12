""" Example 2: Stochastic Gradient Descent """

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm

PATH = "C:/PredML/"
CSV_DATA = "winequality.csv"

dataset  = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,       # Don't include header row as part of data.
                      encoding = "ISO-8859-1", sep=',',
                      names=('fixed acidity', 'volatile acidity', 'citric acid',
                             'residual sugar', 'chlorides', 'free sulfur dioxide',
                             'total sulfur dioxide', 'density', 'pH', 'sulphates',
                             'alcohol', 'quality'))
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(dataset.head())
print(dataset.describe())
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates','alcohol']]

# only significant variables here
X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates','alcohol']]

# Adding an intercept *** This is required ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) # make the predictions by the model
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))


######################### Stochastic Gradient Descent ##################################


print("\nStochastic Gradient Descent")
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Stochastic gradient descent models are sensitive to differences
# in scale so a StandardScaler is usually used.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  # Don't cheat - fit only on training data
X_trainScaled = scaler.transform(X_train)
X_testScaled  = scaler.transform(X_test)

# SkLearn SGD classifier
sgd = SGDRegressor(verbose=1)
sgd.fit(X_trainScaled, y_train)
predictions = sgd.predict(X_testScaled)
print('Root Mean Squared Error:',
      np.sqrt(mean_squared_error(y_test, predictions)))

