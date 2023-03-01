import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import EnsembleVoteClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
# X = df.drop(['Sepsis_Positive'], axis=1)
# X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]
# X = df[['PL', 'M11', 'BD2']]  # --> runner up
X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]  # --> best performing combo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Create classifiers
ada_boost = AdaBoostClassifier()
grad_boost = GradientBoostingClassifier()
xgb_boost = XGBClassifier()
eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting='hard')
# lr = LogisticRegression(fit_intercept=True, solver='liblinear')

# Build array of classifiers.
classifiers = [ada_boost, grad_boost, xgb_boost, eclf]

for clf in classifiers:
    print(clf.__class__.__name__)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)
