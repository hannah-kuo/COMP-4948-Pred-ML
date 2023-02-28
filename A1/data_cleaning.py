import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

# Load the dataset into a pandas DataFrame object
PATH = "C:/PredML/A1/"
CSV_DATA = "Paitients_Files_Train.csv"
df = pd.read_csv(PATH + CSV_DATA, sep=',')

print("\n---------- Raw data ----------\n")
print(df.head(10))  # View a snapshot of the data.
print(df.describe().T)  # View stats including counts which highlight missing values.

# Split the dataset into a target variable and predictor variables
y = df['Sepsis']
X = df.drop(['ID', 'Sepsis'], axis=1)

# Create a KNN imputer object with n_neighbors=5
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in the predictor variables
X_imputed = imputer.fit_transform(X)

# Convert the imputed predictor variables back into a pandas DataFrame object
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Create dummy variables for the 'Sepsis' column
y_dummies = pd.get_dummies(y, prefix='Sepsis').drop(['Sepsis_Negative'], axis=1)

# Combine the imputed predictor variables and dummy target variable into a new DataFrame
df_imputed = pd.concat([X_imputed_df, y_dummies], axis=1)

# Save the cleaned dataset to a new CSV file
df_imputed.to_csv('cleaned_dataset.csv', index=False)

print("\n---------- Cleaned up data ----------\n")
print(df_imputed.head(10))
print(df_imputed.describe().T)
