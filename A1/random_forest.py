import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
X = df.drop(['Sepsis_Positive'], axis=1)
feature_list = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance']

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100)

# Fit the random forest classifier to the data
rf.fit(X, y)

# Extract the feature importances and sort them in descending order
# importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
#
# # Print the feature importances
# print(importances)

# Get numerical feature importances
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


showFeatureImportances(importances, feature_list)
