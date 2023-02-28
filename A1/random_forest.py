import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
X = df.drop(['Sepsis_Positive'], axis=1)

# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Create a random forest classifier object and fit it to the imputed predictor variables and target variable
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_imputed, y)

# Get the feature importances and sort them in descending order
importances = rfc.feature_importances_
indices = importances.argsort()[::-1]

# Print the feature rankings
print("Feature ranking:")
for i in range(X_imputed.shape[1]):
    print("%d. %s (%f)" % (i + 1, X.columns[indices[i]], importances[indices[i]]))
