import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Pie chart of target variable distribution
plt.figure(figsize=(6,6))
df['Sepsis_Positive'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['No Sepsis', 'Sepsis'], colors=['green','red'])
plt.title('Distribution of Sepsis_Positive')
plt.show()

# Histogram of age distribution
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='Age', hue='Sepsis_Positive', kde=True, bins=30, multiple='stack', palette=['green','red'])
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Scatter plot of PL vs M11 colored by Sepsis_Positive
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PL', y='M11', hue='Sepsis_Positive', palette=['green','red'])
plt.title('PL vs M11')
plt.xlabel('PL')
plt.ylabel('M11')
plt.show()

# Box plot of PRG distribution by Sepsis_Positive
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='Sepsis_Positive', y='PRG', palette=['green','red'])
plt.title('Distribution of PRG')
plt.xlabel('Sepsis_Positive')
plt.ylabel('PRG')
plt.show()

# Heatmap of correlation between variables
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.title('Correlation Between Variables')
plt.show()
