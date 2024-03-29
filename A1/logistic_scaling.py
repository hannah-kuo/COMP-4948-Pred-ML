import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']

# best models
X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]  # Average F1: 0.7234200714294805
# X = df[['PL', 'M11', 'BD2', 'PRG']]

# X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]  # Average F1: 0.7107922389186047
# X = df[['PL', 'M11', 'BD2']]
# X = df[['PL', 'M11', 'BD2', 'Age']]
# X = df[['PL', 'M11', 'BD2', 'Insurance']]

# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

""" ADD CROSS FOLD VALIDATION """

print("\n----- LOGISTIC REGRESSION WITH CROSS FOLD VALIDATION -----")

# prepare cross validation with three folds and 1 as a random seed.
kfold = KFold(n_splits=3, shuffle=True)

accuracyList = []
precisionList = []
recallList = []
f1List = []

foldCount = 0

for train_index, test_index in kfold.split(df):
    # use index lists to isolate rows for train and test sets.
    # Get rows filtered by index and all columns.
    # X.loc[row number array, all columns]
    X_train = X_scaled[train_index, :]
    X_test = X_scaled[test_index, :]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Perform logistic regression.
    # logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear', random_state=42)  # Removed random state
    logisticModel = LogisticRegression(max_iter=10000000, random_state=42)
    # Fit the model.
    logisticModel.fit(X_train, y_train.values.ravel())

    y_pred = logisticModel.predict(X_test)
    y_prob = logisticModel.predict_proba(X_test)

    # Show confusion matrix and accuracy scores.
    y_test_array = np.array(y_test)

    print("\n***K-fold: " + str(foldCount))
    foldCount += 1

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average=None, zero_division=1)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1 = metrics.f1_score(y_test, y_pred, average=None)

    accuracyList.append(accuracy)
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)

    print('\nAccuracy: ', accuracy)
    print(classification_report(y_test, y_pred))

    average_precision = average_precision_score(y_test, y_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    # calculate scores
    auc = roc_auc_score(y_test, y_prob[:, 1], )
    print('Logistic: ROC AUC=%.3f' % (auc))

print("\n\nAccuracy, Precision, Recall, F1, and their respective standard deviations For All Folds:")
print("**********************************************************************************************")

print("\nAverage Accuracy: " + str(np.mean(accuracyList)))
print("Accuracy std: " + str(np.std(accuracyList)))

print("\nAverage Precision: " + str(np.mean(precisionList)))
print("Precision std: " + str(np.std(precisionList)))

print("\nAverage Recall: " + str(np.mean(recallList)))
print("Recall std: " + str(np.std(recallList)))

print("\nAverage F1: " + str(np.mean(f1List)))
print("F1 std: " + str(np.std(f1List)))
