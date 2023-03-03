import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

simplefilter("ignore", category=ConvergenceWarning)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
# X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]

X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]  # Average F1 Score: 0.63
# X = df[['PL', 'M11', 'BD2', 'PRG']]  # Average F1 Score: 0.61

# X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]  # Average F1 Score: 0.62
# X = df[['PL', 'M11', 'BD2']]  # Average F1 Score: 0.63
# X = df[['PL', 'M11', 'BD2', 'Insurance']]  # Average F1 Score: 0.62

# Initialize K-Fold cross-validation
k_fold = KFold(n_splits=3, shuffle=True)


def getUnfitModels():
    models = list()

    # models.append(LogisticRegression(max_iter=10000000))
    models.append(LogisticRegression(C=1, penalty='l2'))
    models.append(DecisionTreeClassifier(max_depth=11, min_samples_leaf=2, min_samples_split=10))
    models.append(AdaBoostClassifier())
    # tuned the hyperparameter n_estimators=800 from what was suggested in the best parameters to use for random forest
    models.append(RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=800))

    models.append(GradientBoostingClassifier(learning_rate=0.01, max_depth=7, subsample=1.0))
    models.append(XGBClassifier())
    return models


def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(), 2))
    strStd = str(round(scores.std(), 2))
    print("Mean: " + strMean + "   ", end="")
    print("Std: " + strStd)


def evaluateModel(y_test, predictions, model):
    # Calculate evaluation metrics
    print("\nAverage evaluation metrics over cross fold validation folds:")
    acc_scores = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=k_fold, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=k_fold, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=k_fold, scoring='f1')
    # Print evaluation metrics
    showStats("Accuracy", acc_scores)
    showStats("Precision", precision_scores)
    showStats("Recall", recall_scores)
    showStats("F1 Score", f1_scores)


def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfPredictions[colName] = predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


# Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Get base models.
unfitModels = getUnfitModels()

# --------------------------------------------------------------
# Grid search for optimal hyperparameter tuning
# --------------------------------------------------------------

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
# rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=k_fold, scoring='f1')
#
# # Fit the GridSearchCV object to the data
# rf_grid_search.fit(X_train, y_train)
#
# # Print the best hyperparameters and F1 score for RandomForestClassifier
# print("Best hyperparameters for RandomForestClassifier:", rf_grid_search.best_params_)
# print("Best F1 score for RandomForestClassifier:", rf_grid_search.best_score_)
#
# # Create a GridSearchCV object for XGBClassifier
# xgb_grid_search = GridSearchCV(XGBClassifier(), xgb_param_grid, cv=k_fold, scoring='f1')
#
# # Fit the GridSearchCV object to the data
# xgb_grid_search.fit(X_train, y_train)
#
# # Print the best hyperparameters and F1 score for XGBClassifier
# print("Best hyperparameters for XGBClassifier:", xgb_grid_search.best_params_)
# print("Best F1 score for XGBClassifier:", xgb_grid_search.best_score_)

# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
#
# # Define parameter grid for LogisticRegression
# param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                  'penalty': ['l1', 'l2']}
#
# # Create LogisticRegression model
# model_lr = LogisticRegression(max_iter=100000)
#
# # Perform grid search to find best hyperparameters
# grid_search_lr = GridSearchCV(model_lr, param_grid_lr, cv=k_fold, scoring='f1')
# grid_search_lr.fit(X_train, y_train)
#
# # Print best hyperparameters
# print("Best hyperparameters for LogisticRegression:", grid_search_lr.best_params_)
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.tree import DecisionTreeClassifier
#
# # Define parameter grid for DecisionTreeClassifier
# param_grid_dt = {'max_depth': [3, 5, 7, 9, 11],
#                  'min_samples_split': [2, 5, 10],
#                  'min_samples_leaf': [1, 2, 4]}
#
# # Create DecisionTreeClassifier model
# model_dt = DecisionTreeClassifier()
#
# # Perform grid search to find best hyperparameters
# grid_search_dt = GridSearchCV(model_dt, param_grid_dt, cv=k_fold, scoring='f1')
# grid_search_dt.fit(X_train, y_train)
#
# # Print best hyperparameters
# print("Best hyperparameters for DecisionTreeClassifier:", grid_search_dt.best_params_)

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

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_test, unfitModels)
stackedModel = fitStackedModel(dfPredictions, y_test)

# Evaluate base models with validation data.
print("\n---------- Evaluate Base Models ----------")
for i in range(0, len(models)):
    scores = cross_val_score(models[i], X_test, y_test, cv=5)
    print("\n*** " + models[i].__class__.__name__)
    predictions = models[i].predict(X_test)
    evaluateModel(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
print("\n---------- Evaluate Stacked Model ----------")
scores = cross_val_score(stackedModel, dfPredictions, y_test, cv=5)
stackedPredictions = stackedModel.predict(dfPredictions)
evaluateModel(y_test, stackedPredictions, stackedModel)
