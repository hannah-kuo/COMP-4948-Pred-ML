import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import EnsembleVoteClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
X = df[['PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']]  # --> best performing combo

# Create classifiers
ada_boost = AdaBoostClassifier()
grad_boost = GradientBoostingClassifier()
xgb_boost = XGBClassifier()
eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting='hard')

# Build array of classifiers.
classifiers = [ada_boost, grad_boost, xgb_boost, eclf]

# Set up KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
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
    avg_accuracy = sum(all_accuracy_scores) / len(all_accuracy_scores)
    avg_precision = sum(all_precision_scores) / len(all_precision_scores)
    avg_recall = sum(all_recall_scores) / len(all_recall_scores)
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
    # Print the evaluation metrics for this classifier
    print("Accuracy:", avg_accuracy)
    print("Precision:", avg_precision)
    print("Recall:", avg_recall)
    print("F1-score:", avg_f1)
