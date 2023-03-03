import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import EnsembleVoteClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
# X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]  # --> best performing combo

X = df[['PL', 'M11', 'BD2', 'PRG', 'Age']]

# Scale the predictor variables using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# with tuned hyperparemeter
ada_boost = AdaBoostClassifier(learning_rate=0.1, n_estimators=200)
grad_boost = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, subsample=1.0)
xgb_boost = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=500)
eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting='hard')
lr = LogisticRegression(C=1, penalty='l2', max_iter=100000)

# Build array of classifiers.
classifiers = [ada_boost, grad_boost, xgb_boost, eclf, lr]

# Set up KFold cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Loop through the classifiers and perform cross-validation
for clf in classifiers:
    print()
    print(clf.__class__.__name__)
    # Initialize variables for storing the results across folds
    all_predictions = []
    all_true_labels = []
    all_accuracy_scores = []
    all_precision_scores = []
    all_recall_scores = []
    all_f1_scores = []
    for train_index, test_index in kf.split(X):
        # Split the data into train and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Train the classifier on the training set for this fold
        clf.fit(X_train, y_train)
        # Use the classifier to make predictions on the test set for this fold
        predictions = clf.predict(X_test)
        # Append the predictions and true labels to the running lists
        all_predictions.extend(predictions)
        all_true_labels.extend(y_test)
        # Compute the evaluation metrics for this fold
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        all_accuracy_scores.append(accuracy)
        all_precision_scores.append(precision)
        all_recall_scores.append(recall)
        all_f1_scores.append(f1)

    # Compute the average evaluation metrics across all folds for this classifier
    avg_accuracy = np.mean(all_accuracy_scores)
    std_accuracy = np.std(all_accuracy_scores)
    avg_precision = np.mean(all_precision_scores)
    std_precision = np.std(all_precision_scores)
    avg_recall = np.mean(all_recall_scores)
    std_recall = np.std(all_recall_scores)
    avg_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores)
    # Print the evaluation metrics for this classifier
    print("\nAverage evaluation metrics over cross fold validation folds:")
    print(f"Accuracy:\tmean: {avg_accuracy} \tstd: {std_accuracy}")
    print(f"Precision: \t{avg_precision} \tstd: {std_precision}")
    print(f"Recall: \tmean: {avg_recall} \ttsd: {std_recall}")
    print(f"F1-score: \t{avg_f1} \tstd: {std_f1}")

