""" Example 5: Lasso """


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn                 import metrics
import statsmodels.api       as sm
from sklearn.linear_model import SGDRegressor
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


PATH = "C:/PredML/"
CSV_DATA = "winequality.csv"
dataset  = pd.read_csv(PATH + CSV_DATA)

X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide', 'sulphates',
             'alcohol']]

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X_withConst = sm.add_constant(X)
y = dataset['quality'].values

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm

# Show all columns.
pd.set_option('display.max_columns', None)

# Include only statistically significant columns.
X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide',
             'pH', 'sulphates','alcohol']]
X = sm.add_constant(X)
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Stochastic gradient descent models are sensitive to scaling.
# Fit X scaler and transform X_train.
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
X_train_scaled = scalerX.fit_transform(X_train)

# Build y scaler and transform y_train.
scalerY = StandardScaler()
y_train_scaled = scalerY.fit_transform(np.array(y_train).reshape(-1,1))

# Scale test data.
X_test_scaled = scalerX.transform(X_test)

from sklearn.linear_model import Lasso


def performLassorRegression(X_train, X_test, y_train, y_test, alpha, scalerY):
    lassoreg = Lasso(alpha=alpha)
    lassoreg.fit(X_train, y_train)
    y_pred_scaled = lassoreg.predict(X_test)
    y_pred = scalerY.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))
    print("\n***Lasso Regression Coefficients ** alpha=" + str(alpha))
    print(lassoreg.intercept_)
    print(lassoreg.coef_)
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    
alphaValues = [0, 0.1, 0.5, 1]
for i in range(0, len(alphaValues)):
    performLassorRegression(X_train_scaled, X_test_scaled, y_train_scaled, y_test, alphaValues[i], scalerY)
