import pandas as pd

# URL to source data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Define the column names for the dataset since not provided in the original file
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
           'capital-loss', 'hours-per-week', 'native-country', 'income']

# Fetch the data from the URL
df = pd.read_csv(url, names=columns, sep=',\s*', engine='python')
