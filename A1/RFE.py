import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer

# read the cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# impute missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# define the target variable and predictors
target = 'Sepsis_Positive'
predictors = [col for col in df_imputed.columns if col not in [target, 'ID']]

# create the logistic regression model
lr_model = LogisticRegression(fit_intercept=True, solver='liblinear')

# create the RFE object with 10 features to select
rfe = RFE(estimator=lr_model, n_features_to_select=5)

# fit the RFE object on the imputed dataset
rfe.fit(df_imputed[predictors], df_imputed[target])

# print the selected features
selected_features = [predictors[i] for i in range(len(predictors)) if rfe.support_[i]]
print('Selected features:', selected_features)

