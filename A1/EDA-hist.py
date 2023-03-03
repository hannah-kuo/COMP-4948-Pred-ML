import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Set up the subplots
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))

# Plot the distribution of the target variable
sns.countplot(x='Sepsis_Positive', data=df, ax=axes[0, 0])

# Plot the distribution of each predictor variable
row = 0
col = 1
for idx, col_name in enumerate(df.columns):
    if col_name != 'Sepsis_Positive':
        sns.histplot(data=df, x=col_name, hue='Sepsis_Positive', multiple='stack', ax=axes[row, col])
        if col < 3:
            col += 1
        else:
            col = 0
            row += 1

# Calculate the correlation between each predictor variable and the target variable
correlations = df.corr()['Sepsis_Positive']
print(correlations)

# Show the plots
plt.show()
