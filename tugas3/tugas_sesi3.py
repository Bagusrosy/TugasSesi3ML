import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('Dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
z = dataset.iloc[[2]].values

dataset.info()

dataset.isnull().sum()

dataset.isnull().sum().sum()

print(z)
print(x)
print(y)
'\n'

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)
'\n'

# Encoding categorical data
# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], 
                                                remainder='passthrough')
X = np.array(ct.fit_transform(x))
print(X)
'\n'
# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
'\n'

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1)
print(X_train)
'\n'

print(X_test)
'\n'
print(y_train)
'\n'
print(y_test)
'\n'


# Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
'\n'
print(X_test)
