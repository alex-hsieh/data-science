import pandas as pd
from numpy import nan

# URL to source data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Define the column names for the dataset since not provided in the original file
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
           'capital-loss', 'hours-per-week', 'native-country', 'income']

# Fetch the data from the URL
df = pd.read_csv(url, names=columns, sep=',\s*', engine='python')

# stripping whitespace from string columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# handling the missing values represented by '?'
df = df.replace('?', nan)
df.isnull().sum()
df.dropna(inplace=True)

# encoding the target variable 'income' to binary values
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# dropping redudnant columns
df.drop(['fnlwgt', 'education'], axis=1, inplace=True)