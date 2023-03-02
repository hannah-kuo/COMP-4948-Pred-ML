import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Plot the distribution of the target variable
sns.countplot(x='Sepsis_Positive', data=df)
plt.show()

# Plot the distribution of each predictor variable
for col in df.columns:
    if col != 'Sepsis_Positive':
        sns.histplot(data=df, x=col, hue='Sepsis_Positive', multiple='stack')
        plt.show()

# Calculate the correlation between each predictor variable and the target variable
correlations = df.corr()['Sepsis_Positive']
print(correlations)
