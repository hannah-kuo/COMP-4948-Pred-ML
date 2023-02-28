import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
X = df.drop(['Sepsis_Positive'], axis=1)

# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Convert the imputed predictor variables back into a pandas DataFrame object
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Create a random forest classifier object
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Create an RFE object with 5-fold cross-validation
rfe = RFECV(estimator=rfc, step=1, cv=5, scoring='accuracy')

# Fit the RFE object to the imputed predictor variables and target variable
rfe.fit(X_imputed_df, y)

# Print the selected features and their rankings
selected_features = X_imputed_df.columns[rfe.support_]
feature_rankings = rfe.ranking_
print('Selected features:', list(selected_features))
print('Feature rankings:', list(feature_rankings))
