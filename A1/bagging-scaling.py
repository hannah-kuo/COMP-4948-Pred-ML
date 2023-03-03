import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
# X = df.drop(['Sepsis_Positive'], axis=1)
X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]
# X = df[['PL', 'M11', 'BD2', 'PRG']]

# X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]
# X = df[['PL', 'M11', 'BD2']]
# X = df[['PL', 'M11', 'BD2', 'Age']]
# X = df[['PL', 'M11', 'BD2', 'Insurance']]

# Create classifiers
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
# lr = LogisticRegression(fit_intercept=True, solver='liblinear')
lr = LogisticRegression(max_iter=10000000, random_state=42)

# Build array of classifiers.
classifierArray = [knn, svc, rg, lr]

# Initialize K-Fold cross-validation
k_fold = KFold(n_splits=3, shuffle=True)


def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(), 2))
    strStd = str(round(scores.std(), 2))
    print("Mean: " + strMean + "   ", end="")
    print("Std: " + strStd)


# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)


def evaluateModel(model, X_test, y_test, title):
    print("\n\n*** " + title + " ***")
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    # print(report)
    # Calculate evaluation metrics
    print("-- Average evaluation metrics over cross fold validation folds --")
    acc_scores = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=k_fold, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=k_fold, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=k_fold, scoring='f1')
    # Print evaluation metrics
    showStats("Accuracy", acc_scores)
    showStats("Precision", precision_scores)
    showStats("Recall", recall_scores)
    showStats("F1 Score", f1_scores)


# Search for the best classifier.
for clf in classifierArray:
    modelType = clf.__class__.__name__

    # Create and evaluate stand-alone model.
    clfModel = clf.fit(X_train, y_train)
    evaluateModel(clfModel, X_test, y_test, modelType)

    # max_features means the maximum number of features to draw from X.
    # max_samples sets the percentage of available data used for fitting.
    # did hyperparameter tuning for "max_features" and "n_estimators"
    # max_samples = 0.4, 0.1
    bagging_clf = BaggingClassifier(clf, max_samples=0.1, max_features=5, n_estimators=1000)
    baggedModel = bagging_clf.fit(X_train, y_train)
    evaluateModel(baggedModel, X_test, y_test, "BAGGED: " + modelType)
