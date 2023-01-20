from sklearn import datasets
import numpy as np
from sklearn import datasets
from   sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset.
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
iris['target_names']
print(data.head())

# Import train_test_split function
from sklearn.model_selection import train_test_split
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Saving feature names for later use
feature_list = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc_x            = StandardScaler()
X_train_scaled  = sc_x.fit_transform(X_train)
X_test_scaled = sc_x.transform(X_test)

from sklearn              import metrics
from sklearn.ensemble     import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def buildModelAndPredict(clf, X_train_scaled, X_test_scaled, y_train, y_test, title):
    print("\n********\n " + title)
    #Train the model using the training sets y_pred=rf.predict(X_test)
    clf_fit = clf.fit(X_train_scaled,y_train)
    y_pred = clf_fit.predict(X_test_scaled)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # For explanation see:
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
    print(metrics.classification_report(y_test, y_pred, digits=3))

    # Predict species for a single flower.
    # sepal length = 3, sepal width = 5
    # petal length = 4, petal width = 2
    prediction = clf_fit.predict([[3, 5, 4, 2]])

    # 'setosa', 'versicolor', 'virginica'
    print(prediction)


lr = LogisticRegression(fit_intercept=True, solver='liblinear')
buildModelAndPredict(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")


""" Adding Random Forest Model """

print("\n*********\nRandom Forest Regression:")

#Create a Gaussian Classifier
rf = RandomForestClassifier(n_estimators=200)


#Train the model using the training sets y_pred=rf.predict(X_test)
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')

# Print out the mean square error.
mse = mean_squared_error(y_test, predictions)
print('RMSE:', np.sqrt(mse))

# Predict species for a single flower.
# sepal length = 3, sepal width = 5
# petal length = 4, petal width = 2
prediction = rf.predict([[3, 5, 4, 2]])
# 'setosa', 'versicolor', 'virginica'
print(prediction)


""" Code for displaying feature importance """

# Get numerical feature importances
importances = list(rf.feature_importances_)


# Present features and importance scores.
def showFeatureImportances(importances, feature_list):
    dfImportance = pd.DataFrame()
    for i in range(0, len(importances)):
        dfImportance = dfImportance.append({"importance":importances[i],
                                            "feature":feature_list[i] },
                                            ignore_index = True)

    dfImportance = dfImportance.sort_values(by=['importance'],
                                            ascending=False)
    print(dfImportance)


showFeatureImportances(importances, feature_list)


""" Code to build forest with important features only """


print("\n**********************************\nBuilding forest with important features only:")
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators=1000, random_state=42)

# Extract the two most important features
important_indices = ['petal length', 'petal width', 'sepal length']
train_important = X_train[important_indices]
test_important = X_test[important_indices]

# Train the random forest
rf_most_important.fit(train_important, y_train)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - y_test)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / y_test))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

buildModelAndPredict(rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest Regression")
