import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
# X = df.drop(['Sepsis_Positive'], axis=1)
# X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]
X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]  # --> this yielded slightly better f-1 scores


# Create classifiers
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
lr = LogisticRegression(fit_intercept=True, solver='liblinear')

# Build array of classifiers.
classifierArray = [knn, svc, rg, lr]


def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(), 2))

    strStd = str(round(scores.std(), 2))
    print("Mean: " + strMean + "   ", end="")
    print("Std: " + strStd)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def evaluateModel(model, X_test, y_test, title):
    print("\n*** " + title + " ***")
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)


# Search for the best classifier.
for clf in classifierArray:
    modelType = clf.__class__.__name__

    # Create and evaluate stand-alone model.
    clfModel = clf.fit(X_train, y_train)
    evaluateModel(clfModel, X_test, y_test, modelType)

    # max_features means the maximum number of features to draw from X.
    # max_samples sets the percentage of available data used for fitting.
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=6, n_estimators=100)
    baggedModel = bagging_clf.fit(X_train, y_train)
    evaluateModel(baggedModel, X_test, y_test, "BAGGED: " + modelType)
