from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import average_precision_score
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import warnings

from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)

""" Data Cleaning """

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

# Load the dataset into a pandas DataFrame object
PATH = "C:/PredML/A1/"
CSV_DATA = "Paitients_Files_Train.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')

print("\n---------- Raw data ----------\n")
print(df.head(10))  # View a snapshot of the data.
print()
print(df.describe().T)  # View stats including counts which highlight missing values.

# Split the dataset into a target variable and predictor variables
y = df['Sepsis']
X = df.drop(['ID', 'Sepsis'], axis=1)

# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Convert the imputed predictor variables back into a pandas DataFrame object
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Create dummy variables for the 'Sepsis' column
y_dummies = pd.get_dummies(y, prefix='Sepsis').drop(['Sepsis_Negative'], axis=1)

# Combine the imputed predictor variables and dummy target variable into a new DataFrame
df = pd.concat([X_imputed_df, y_dummies], axis=1)

print("\n---------- Cleaned up data ----------\n")
print(df.head(10))
print()
print(df.describe().T)

""" Feature Selection - Recursive Feature Elimination (RFE) """

# impute missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# define the target variable and predictors
target = 'Sepsis_Positive'
predictors = [col for col in df_imputed.columns if col not in [target, 'ID']]

# create the logistic regression model with 'l2' penalty
lr_model = LogisticRegression(max_iter=10000000, random_state=42)

# create the RFE object with 10 features to select
rfe = RFE(estimator=lr_model, n_features_to_select=5)

# fit the RFE object on the imputed dataset
rfe.fit(df_imputed[predictors], df_imputed[target])

# print the selected features
selected_features = [predictors[i] for i in range(len(predictors)) if rfe.support_[i]]
print("\n\n*** Recursive Feature Elimination")
print('\nSelected features:', selected_features)

""" Feature Selection - Forward Feature Selection (FFS) """

# impute missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# split data into features and target variable
X = df_imputed.drop(['Sepsis_Positive'], axis=1)
y = df_imputed['Sepsis_Positive']

# Perform logistic regression.
lr_model = logisticModel = LogisticRegression(max_iter=10000000, random_state=42)

# forward feature selection with f_regression
# f_regression returns F statistic for each feature.
ffs = f_regression(X, y)

featuresDf = pd.DataFrame()
for i in range(0, len(X.columns)):
    featuresDf = featuresDf.append({"feature": X.columns[i],
                                    "ffs": ffs[0][i]}, ignore_index=True)
featuresDf = featuresDf.sort_values(by=['ffs'], ascending=False)
print("\n\n*** Forward Feature Selection")
print("\nSignificant features in descending F-statistic values:")
print(featuresDf)


""" Feature Selection - Random Forest """


# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
X = df.drop(['Sepsis_Positive'], axis=1)
feature_list = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)

# Use the random grid to search for best hyperparameters
rf = RandomForestRegressor()
random_grid = \
    {'bootstrap': [True],
     'max_depth': [4, 6, None],
     'max_features': [1.0],
     'min_samples_leaf': [15],
     'min_samples_split': [15],
     'n_estimators': [400, 800, 1600]}

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=9, cv=3, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print("\n\n*** Random Forest Feature Selection")
print("\nBest parameters to use for random forest")
print(rf_random.best_params_)
print("----------------------------------------")

# We create the new rf with the best random forest params suggested above
rf = RandomForestRegressor(n_estimators=800, min_samples_split=15, min_samples_leaf=15, max_features='auto',
                           max_depth=6, bootstrap=True)
rf.fit(X_train, y_train)

# Find feature importance's
importances = list(rf.feature_importances_)


# Present features and importance scores.
def showFeatureImportances(importances, feature_list):
    dfImportance = pd.DataFrame()
    for i in range(0, len(importances)):
        dfImportance = dfImportance.append({"importance": importances[i],
                                            "feature": feature_list[i]},
                                           ignore_index=True)

    dfImportance = dfImportance.sort_values(by=['importance'],
                                            ascending=False)
    print(dfImportance)


showFeatureImportances(importances, X_train.columns)

""" Logistic Regression with Cross Fold Validation """

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']

# chosen variables based on feature selection
X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]

# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

print("\n\n\n----- Logistic Regression with Cross Fold Validation and Scaling -----")

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
    X_train = X.iloc[train_index, :]
    X_test = X.iloc[test_index, :]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, solver='lbfgs', penalty='l2')
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

""" Finding the best hyper-parameters for the classifiers/models later """

# --------------------------------------------------------------
# Grid search for optimal hyperparameter tuning
# --------------------------------------------------------------

# Initialize K-Fold cross-validation
k_fold = KFold(n_splits=3, shuffle=True)

# Define the hyperparameter grid for RandomForestClassifier
rf_param_grid = {
    'n_estimators': [800],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define the hyperparameter grid for XGBClassifier
xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.8, 1.0]
}

# Create a GridSearchCV object for RandomForestClassifier
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=k_fold, scoring='f1')

# Fit the GridSearchCV object to the data
rf_grid_search.fit(X_train, y_train)

# Print the best hyperparameters and F1 score for RandomForestClassifier
print("Best hyperparameters for RandomForestClassifier:", rf_grid_search.best_params_)
print("Best F1 score for RandomForestClassifier:", rf_grid_search.best_score_)

# Create a GridSearchCV object for XGBClassifier
xgb_grid_search = GridSearchCV(XGBClassifier(), xgb_param_grid, cv=k_fold, scoring='f1', error_score='raise')

# Fit the GridSearchCV object to the data
xgb_grid_search.fit(X_train, y_train)

# Print the best hyperparameters and F1 score for XGBClassifier
print("Best hyperparameters for XGBClassifier:", xgb_grid_search.best_params_)
print("Best F1 score for XGBClassifier:", xgb_grid_search.best_score_)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define parameter grid for LogisticRegression
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                 'penalty': ['l1', 'l2']}

# create the logistic regression model with 'l2' penalty
model_lr = LogisticRegression(fit_intercept=True, solver='lbfgs', penalty='l2')

# Perform grid search to find best hyperparameters
grid_search_lr = GridSearchCV(model_lr, param_grid_lr, cv=k_fold, scoring='f1')
grid_search_lr.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters for LogisticRegression:", grid_search_lr.best_params_)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define parameter grid for DecisionTreeClassifier
param_grid_dt = {'max_depth': [3, 5, 7, 9, 11],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}

# Create DecisionTreeClassifier model
model_dt = DecisionTreeClassifier()

# Perform grid search to find best hyperparameters
grid_search_dt = GridSearchCV(model_dt, param_grid_dt, cv=k_fold, scoring='f1')
grid_search_dt.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters for DecisionTreeClassifier:", grid_search_dt.best_params_)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Define parameter grid for AdaBoostClassifier
param_grid_ab = {'n_estimators': [50, 100, 200],
                 'learning_rate': [0.01, 0.1, 1],
                 'base_estimator__max_depth': [1, 3, 5]}

# Create base estimator for AdaBoostClassifier
base_estimator_ab = DecisionTreeClassifier()

# Create AdaBoostClassifier model
model_ab = AdaBoostClassifier(base_estimator=base_estimator_ab)

# Perform grid search to find best hyperparameters
grid_search_ab = GridSearchCV(model_ab, param_grid_ab, cv=k_fold, scoring='f1')
grid_search_ab.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters for AdaBoostClassifier:", grid_search_ab.best_params_)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 1],
}

# Create an instance of the XGBClassifier
xgb = XGBClassifier()

# Create the GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='f1')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print(f"Best hyperparameters for XGBClassifier: {grid_search.best_params_}")

""" Bagged Model with Cross Fold Validation, Hyperparameter Tuning, and Scaling"""

""" Ensemble Model with Cross Fold Validation, Hyperparameter Tuning, and Scaling"""

""" Stacked Model with Cross Fold Validation, Hyperparameter Tuning, and Scaling"""
