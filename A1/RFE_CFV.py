import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import f_regression, RFE
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None

# read the cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# impute missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split data into features and target variable
X = df_imputed.drop(['Sepsis_Positive'], axis=1)
y = df_imputed['Sepsis_Positive']

# Initialize logistic regression model and RFE selector
logreg = LogisticRegression(max_iter=10000)
selector = RFE(logreg)

# Set up cross-validation
cv = KFold(n_splits=3, shuffle=True)

accuracyList = []
precisionList = []
recallList = []
f1List = []

foldCount = 0

# Perform feature selection with cross-validation
for train_index, test_index in cv.split(df_imputed):
    # use index lists to isolate rows for train and test sets.
    # Get rows filtered by index and all columns.
    # X.loc[row number array, all columns]
    X_train = X.loc[train_index, :]
    X_test = X.loc[test_index, :]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')  # Removed random state
    # Fit the model.
    logisticModel.fit(X_train, y_train.values.ravel())

    y_pred = logisticModel.predict(X_test)
    y_prob = logisticModel.predict_proba(X_test)

    # Show confusion matrix and accuracy scores.
    y_test_array = np.array(y_test)
    cm = pd.crosstab(y_test_array, y_pred, rownames=['Actual'], colnames=['Sepsis_Positive'])

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
    print("\nConfusion Matrix")
    print(cm)
    print(classification_report(y_test, y_pred, zero_division=1))

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
