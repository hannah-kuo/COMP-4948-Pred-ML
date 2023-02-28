import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score

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

# Create a sequential feature selector object with forward selection and 5-fold cross-validation
sfs = SequentialFeatureSelector(rfc, n_features_to_select=5, direction='forward', cv=5)

# Fit the sequential feature selector object to the imputed predictor variables and target variable
sfs.fit(X_imputed_df, y)

# Print the selected features and their cross-validation scores
selected_features = X_imputed_df.columns[sfs.get_support()]
cv_scores = cross_val_score(rfc, X_imputed_df[selected_features], y, cv=5)
print('Selected features:', list(selected_features))
print('Cross-validation scores:', list(cv_scores))
