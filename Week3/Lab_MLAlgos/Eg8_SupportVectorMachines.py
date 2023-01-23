""" Example 8: Support Vector Machines with a Linear Kernel """


from sklearn.linear_model import ElasticNet

bestRMSE = 100000.03

""" Example 5: Lasso """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from sklearn.linear_model import SGDRegressor
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

PATH = "C:/PredML/"
CSV_DATA = "winequality.csv"
dataset = pd.read_csv(PATH + CSV_DATA)

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
import statsmodels.api as sm

# Show all columns.
pd.set_option('display.max_columns', None)

# Include only statistically significant columns.
X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide',
             'pH', 'sulphates', 'alcohol']]
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
y_train_scaled = scalerY.fit_transform(np.array(y_train).reshape(-1, 1))

# Scale test data.
X_test_scaled = scalerX.transform(X_test)

# Import svm package
from sklearn import svm

# Create a svm Classifier using one of the following options:
# linear, polynomial, and radial
clf = svm.SVC(kernel='linear')

# Train the model using the training set.
clf.fit(X_train, y_train)

# Evaluate the model.
y_pred = clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
