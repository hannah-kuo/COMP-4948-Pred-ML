import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from torch import nn
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('C:/PredML/wildlife.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network architecture
class AnimalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AnimalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set hyperparameters, loss function, and optimizer
input_dim = X_train_tensor.shape[1]
hidden_dim = 64
output_dim = y_train_tensor.shape[1]
learning_rate = 0.001
num_epochs = 100

model = AnimalClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model and visualize progress
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = torch.sigmoid(y_pred)
    y_pred_classes = (y_pred > 0.5).numpy()
    accuracy = np.mean(np.all(y_pred_classes == y_test, axis=1))
    print(f'Accuracy: {accuracy * 100:.2f}%')

    precision = precision_score(y_test, y_pred_classes, average='macro')
    recall = recall_score(y_test, y_pred_classes, average='macro')
    f1 = f1_score(y_test, y_pred_classes, average='macro')

    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1-score: {f1 * 100:.2f}%')


