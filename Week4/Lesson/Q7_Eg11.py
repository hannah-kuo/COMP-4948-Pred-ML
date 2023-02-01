""" Example 11: Grid Searching the Learning Rate """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score, classification_report

PATH = "C:/PredML/"
FILE = "Social_Network_Ads.csv"
data = pd.read_csv(PATH + FILE)
y = data["Purchased"]
X = data.copy()
del X['User ID']
del X['Purchased']
X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Stochastic gradient descent models are sensitive to differences
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_trainScaled = scaler.transform(X_train)
X_testScaled = scaler.transform(X_test)
X_valScaled = scaler.transform(X_val)


def showResults(networkStats):
    dfStats = pd.DataFrame.from_records(networkStats)
    dfStats = dfStats.sort_values(by=['f1'])
    print(dfStats)


def evaluate_model(predictions, y_test):
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("Precision: " + str(precision) + " " + \
          "Recall: " + str(recall) + " " + \
          "F1: " + str(f1))
    return precision, recall, f1


print("\nLogistic Regression:")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_trainScaled, y_train)
predictions = clf.predict(X_testScaled)
evaluate_model(predictions, y_test)

COLUMN_DIMENSION = 1
# --------------------------------------------------------------
# Part 2
from keras.models import Sequential
from keras.layers import Dense

# shape() obtains rows (dim=0) and columns (dim=1)
n_features = X_trainScaled.shape[COLUMN_DIMENSION]


def getPredictions(model, X_test):
    probabilities = model.predict(X_test)

    predictions = []
    for i in range(len(probabilities)):
        if (probabilities[i][0] > 0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


networkStats = []

# -------------------Model parameters---------------------------

neuronList = [5, 25, 50, 100, 150]

# --------------------------------------------------------------

# ------------ Build model -------------------------------------
# Build model
import keras
from keras.optimizers import Adam  # for adam optimizer


def create_model(numNeurons):
    model = Sequential()
    model.add(Dense(numNeurons,
                    input_dim=13, kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform'))

    # Use Adam optimizer with the given learning rate
    LEARNING_RATE = 0.0100
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


for numNeurons in neuronList:
    BATCH_SIZE = 10
    EPOCHS = 100
    model = create_model(numNeurons)
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, verbose=1,
                        validation_data=(X_val, y_val))
    rmse = evaluate_model(model, X_test, y_test)
    networkStats.append({"rmse": rmse, "# neurons": numNeurons})
showResults(networkStats)

# --------------------------------------------------------------
