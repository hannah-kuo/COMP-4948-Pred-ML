import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Dense
import numpy as np

PATH = "C:/PredML/"
df = pd.read_csv(PATH + 'bill_authentication.csv')

# Convert text to numeric category.
# 0 is setosa, 1 is versacolor and 2 is virginica
y = df['Class']
X = df
del X['Class']
ROW_DIM = 0
COL_DIM = 1

# Create vertical array of features.
x_array = X.values
x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
                                  x_array.shape[COL_DIM])

y_array = np.array(y.values)
y_arrayReshaped = y_array.reshape(len(y_array), 1)


# Split into train, validation and test data sets.
X_train, X_temp, y_train, y_temp = train_test_split(
    x_arrayReshaped, y_arrayReshaped, test_size=0.33)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50)

n_features = X_train.shape[COL_DIM]


# Define the model.
model = Sequential()

# Hidden layer 1 (also receives the input layer)
model.add(Dense(2, activation='relu', input_shape=(n_features,)))

# Output layer
model.add(Dense(3, activation='softmax'))

# Compile the model.
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model.
history = model.fit(X_train, y_train, epochs=1000, batch_size=28, verbose=1,
                    validation_data=(X_val, y_val))

# Evaluate the model with unseen data.
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# make a prediction
row = [5.1, 3.5, 1.4, 0.2]
yhat = model.predict([row])
print('Predicted: s (class=d)' + str(yhat))

import matplotlib.pyplot as plt
