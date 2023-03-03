import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
# X = df.drop(['Sepsis_Positive'], axis=1)
# X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]

# the next two perform pretty similarly
X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]
# X = df[['PL', 'M11', 'BD2']]

def getUnfitModels():
    models = list()
    models.append(LogisticRegression())
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(RandomForestClassifier(n_estimators=100))
    return models


def evaluateModel(y_test, predictions, model):
    print("\n*** " + model.__class__.__name__)
    report = classification_report(y_test, predictions)
    print(report)


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_test, unfitModels)
stackedModel = fitStackedModel(dfPredictions, y_test)

# Evaluate base models with validation data.
print("\n---------- Evaluate Base Models ----------")
for i in range(0, len(models)):
    scores = cross_val_score(models[i], X_test, y_test, cv=5)
    print("\n*** " + models[i].__class__.__name__)
    print("Accuracy:", round(scores.mean(), 3))
    predictions = models[i].predict(X_test)
    evaluateModel(y_test, predictions, models[i])

# Evaluate stacked model with validation data.
print("\n---------- Evaluate Stacked Model ----------")
scores = cross_val_score(stackedModel, dfPredictions, y_test, cv=5)
print("Accuracy:", round(scores.mean(), 3))
stackedPredictions = stackedModel.predict(dfPredictions)
evaluateModel(y_test, stackedPredictions, stackedModel)
