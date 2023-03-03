import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Create a figure and subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))

# Loop through each predictor variable and create a boxplot
for ax, col in zip(axes.flatten(), df.columns[:-1]):
    ax.boxplot(df[col])
    ax.set_title(col)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Value')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
