import pandas as pd
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# URL to source data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Define the column names for the dataset since not provided in the original file
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 
           'capital-loss', 'hours-per-week', 'native-country', 'income']

# Fetch the data from the URL
df = pd.read_csv(url, names=columns, sep=r',\s*', engine='python')

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

# one-hot encoding for categorical variables
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 
                    'race', 'sex', 'native-country']
df = pd.get_dummies(df, columns=categorical_cols)

# create target vector and feature matrix
y = df['income']
X = df.drop('income', axis=1)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# test knn 
k_list = [1, 3, 5, 7, 9, 11, 15, 20]
accuracies = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    accuracies.append(score)
    print(f"k={k}, Accuracy={score}")

# Use the best k from your output
best_k = 20
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)
y_pred = knn_final.predict(X_test_scaled)

# Print the required metrics
print(f"\nFinal Results for k={best_k}:")
print(classification_report(y_test, y_pred))

# Plot 1: Accuracy vs k
plt.figure(figsize=(10,6))
plt.plot(k_list, accuracies, marker='o', linestyle='dashed')
plt.title('KNN Accuracy vs. k Value')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# Plot 2: Confusion Matrix Heatmap
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for k={best_k}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()