import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import accuracy_score

# read the cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# impute missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# split data into features and target variable
X = df_imputed.drop(['Sepsis_Positive'], axis=1)
y = df_imputed['Sepsis_Positive']

# initialize logistic regression model
lr = LogisticRegression(max_iter=10000)

# forward feature selection with f_regression
features = []
scores = []
for i in range(len(X.columns)):
    best_score = 0
    best_feature = ""
    for col in X.columns:
        if col not in features:
            new_features = features + [col]
            X_new = X[new_features]
            # f_regression returns F statistic for each feature.
            f_scores, _ = f_regression(X_new, y)
            lr.fit(X_new, y)
            y_pred = lr.predict(X_new)
            score = accuracy_score(y, y_pred) * f_scores[i]
            if score > best_score:
                best_score = score
                best_feature = col
    features.append(best_feature)
    scores.append(best_score)
    print("Added feature:", best_feature, "Score:", best_score)

# print selected features and their scores
print("Selected Features:", features)
print("Feature Scores:", scores)
