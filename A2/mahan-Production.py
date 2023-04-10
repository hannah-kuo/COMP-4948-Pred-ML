import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from keras.models import load_model

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def evaluateModel(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse), 3)
    mae = round(mean_absolute_error(y_test, predictions), 3)
    r2 = round(r2_score(y_test, predictions), 3)

    print("Model: " + model.__class__.__name__)
    print("RMSE: " + str(rmse) + ", MAE: " + str(mae) + ", R2: " + str(r2) + ", MSE: " + str(mse))



def remove_outlier(df, col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    df_out = df.loc[(df[col_name] > Q1 - 1.5 * IQR) & (df[col_name] < Q3 + 1.5 * IQR)]
    return df_out


def load_saved_model(file_name):
    return load_model(file_name)


def load_saved_scaler(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# Load and prepare test data
# test_path = Path("test.csv")
# PATH = Path("/Users/mahan/Desktop/Winter2023/Predictive-Machine- 4948/DataSets/car_purchasing.csv")

test_df = pd.read_csv("C:/PredML/A2/mahan/test.csv", encoding="ISO-8859-1", sep=',')
# test_df.drop(columns=['car purchase amount'], inplace=True)
test_df = pd.get_dummies(test_df, columns=['gender'])

test_df = remove_outlier(test_df, 'age')
test_df = remove_outlier(test_df, 'net worth')
test_df = remove_outlier(test_df, 'annual Salary')

X_test = test_df[['age', 'annual Salary', 'net worth']]
Y_test = test_df['car purchase amount']


# Load saved scalers
def load_saved_scaler(file_name):
    return joblib.load(file_name)


# Load saved scalers
scaler_filename = "BinaryFolder/scaler.pkl"
scaler = load_saved_scaler(scaler_filename)
# X_test_scaled = scaler.transform(X_test)
X_test_scaled = X_test
# Load saved base models
base_model_filenames = [
    "base_model_0.pkl",
    "base_model_1.pkl",
    "base_model_2.pkl",
    "base_model_3.pkl",
    "base_model_4.pkl",
    "base_model_5.pkl",
    "basic_sequential_NN_model.pkl",
    "sequential_NN_model.pkl",
]
base_models = []

for filename in base_model_filenames:
    if filename == "sequential_NN_model.pkl":

        # Load the saved model
        # Load the saved model
        sequential_NN_model = load_model('BinaryFolder/sequential_NN_model.pkl')
        base_models.append(sequential_NN_model)
    elif filename == "basic_sequential_NN_model.pkl":
        from keras.models import load_model

        # Load the saved model
        sequential_NN_model = load_model('BinaryFolder/basic_sequential_NN_model.pkl')
        base_models.append(sequential_NN_model)
    else:
        model = joblib.load("BinaryFolder/"+filename)
        base_models.append(model)
        print("model type:", filename, type(model))

print()

# Get base model predictions
df_base_predictions = pd.DataFrame()
from keras.engine.sequential import Sequential
import statsmodels.api as sm

for i, model in enumerate(base_models):
    # if isinstance(model, Sequential):
    #     X_test_scaled = sm.add_constant(X_test_scaled)
    if hasattr(model, 'predict'):
        predictions = model.predict(X_test_scaled)
    else:
        predictions = model.predict(X_test_scaled)
    df_base_predictions[f"model_{i + 1}"] = predictions

# Load the stacked model
# stacked_model_filename = "stacked_model.pkl"
# stacked_model = load_model('stacked_model.pkl')
from joblib import dump, load

# Save the model to a file
# dump(stacked_model, 'stacked_model.pkl')

# Load the model from the file
stacked_model = load('BinaryFolder/stacked_model.pkl')

# with open(stacked_model_filename, 'rb') as f:
#     stacked_model = pickle.load(f)

# Make predictions with the stacked model
stacked_predictions = stacked_model.predict(df_base_predictions)
# print("Stacked Predictions:")
# print(stacked_predictions)
print(stacked_predictions)

# Store predictions in a dataframe
dfPredictions = pd.DataFrame()
listPredictions = []
for i in range(0, len(stacked_predictions)):
    prediction = stacked_predictions[i]
    listPredictions.append(prediction)
dfPredictions['Predictions'] = listPredictions

dfPredictions.to_csv('Predictions.csv', index=False)

print("\n** Evaluate Stacked Model **")
evaluateModel(Y_test, stacked_predictions, stacked_model)
