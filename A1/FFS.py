import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_regression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
# f_regression returns F statistic for each feature.
ffs = f_regression(X, y)

featuresDf = pd.DataFrame()
for i in range(0, len(X.columns)):
    featuresDf = featuresDf.append({"feature": X.columns[i],
                                    "ffs": ffs[0][i]}, ignore_index=True)
featuresDf = featuresDf.sort_values(by=['ffs'], ascending=False)
print("\nSignificant features in descending F-statistic values:")
print(featuresDf)
