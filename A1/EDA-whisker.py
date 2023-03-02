import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into a target variable and predictor variables
y = df['Sepsis_Positive']
X = df.drop(['Sepsis_Positive'], axis=1)

# Plot distribution of target variable
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Distribution of Target Variable')
plt.xlabel('Sepsis_Positive')
plt.ylabel('Count')
plt.show()

# Plot boxplot of each predictor variable
for col in X.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=y, y=X[col])
    plt.title(f'{col} vs. Sepsis_Positive')
    plt.xlabel('Sepsis_Positive')
    plt.ylabel(col)
    plt.show()
