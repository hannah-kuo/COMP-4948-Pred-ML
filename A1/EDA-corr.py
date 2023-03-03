import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Select variables to plot
# variables = ['Sepsis_Positive', 'PL', 'M11', 'BD2', 'PRG', 'Age', 'Insurance']

# Create pair plot
# sns.pairplot(df[variables], corner=True)
sns.pairplot(df)

# Show the plot
plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
#
# df = pd.read_csv('cleaned_dataset.csv')
#
# sns.set(style="ticks", color_codes=True)
# g = sns.pairplot(df)
# plt.show()

