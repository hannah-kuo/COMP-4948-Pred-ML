import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset into a pandas DataFrame object
df = pd.read_csv('cleaned_dataset.csv')

# Age vs Sepsis_Positive - bar chart
age_groups = pd.cut(df['Age'], bins=[0, 18, 30, 40, 50, 60, 70, 80, 90, 100])
age_sepsis = pd.crosstab(age_groups, df['Sepsis_Positive'])
ax = age_sepsis.plot(kind='bar', stacked=True)
ax.set_xlabel('Age Group')
ax.set_ylabel('Number of Patients')
ax.set_title('Age vs Sepsis_Positive')
plt.gcf().subplots_adjust(bottom=0.20)

# PL vs Sepsis_Positive - histogram
plt.figure()
sns.histplot(data=df, x='PL', hue='Sepsis_Positive', element='step', kde=True)
plt.xlabel('PL Value')
plt.ylabel('Frequency')
plt.title('PL vs Sepsis_Positive')

# M11 vs Sepsis_Positive - box plot
plt.figure()
sns.boxplot(x='Sepsis_Positive', y='M11', data=df)
plt.xlabel('Sepsis_Positive')
plt.ylabel('M11 Value')
plt.title('M11 vs Sepsis_Positive')

# BD2 vs Sepsis_Positive - scatter plot
plt.figure()
sns.scatterplot(x='BD2', y='Sepsis_Positive', data=df)
plt.xlabel('BD2 Value')
plt.ylabel('Sepsis_Positive')
plt.title('BD2 vs Sepsis_Positive')

# PRG vs Sepsis_Positive - bar chart
prg_sepsis = pd.crosstab(df['PRG'], df['Sepsis_Positive'])
prg_sepsis.plot(kind='bar', stacked=True)
plt.xlabel('PRG Value')
plt.ylabel('Number of Patients')
plt.title('PRG vs Sepsis_Positive')
plt.gcf().subplots_adjust(bottom=0.15)

# Insurance vs Sepsis_Positive - stacked bar chart
ins_sepsis = pd.crosstab(df['Insurance'], df['Sepsis_Positive'])
ins_sepsis.plot(kind='bar', stacked=True)
plt.xlabel('Insurance Type')
plt.ylabel('Number of Patients')
plt.title('Insurance vs Sepsis_Positive')
plt.gcf().subplots_adjust(bottom=0.15)

# Show all plots
plt.show()
