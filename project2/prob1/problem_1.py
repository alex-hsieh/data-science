from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 
  
# metadata 
print(car_evaluation.metadata) 
  
# variable information 
print(car_evaluation.variables) 

# first 5 rows of the data
print(X.head())

# size and shape of the data
print(X.shape)

# missing values and attributes in the data
print(X.info())
print(X.isnull().sum())

# column names and data types
buying_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
maint_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
doors_mapping = {'2': 2, '3': 3, '4': 4, '5more': 5}
persons_mapping = {'2': 2, '4': 4, 'more': 5}
lug_boot_mapping = {'small': 1, 'med': 2, 'big': 3}
safety_mapping = {'low': 1, 'med': 2, 'high': 3}

# mapping the categorical variables to numerical values
X['buying'] = X['buying'].map(buying_mapping)
X['maint'] = X['maint'].map(maint_mapping)
X['doors'] = X['doors'].map(doors_mapping)
X['persons'] = X['persons'].map(persons_mapping)
X['lug_boot'] = X['lug_boot'].map(lug_boot_mapping)
X['safety'] = X['safety'].map(safety_mapping)

# scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
