# Step 1: Importing the libraries
import numpy as np
import pandas as pd

# Step 2: Importing dataset
dataset = pd.read_csv('../datasets/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#-----some expansions
# a[-1]    # last item in the array
# a[-2:]   # last two items in the array
# a[:-2]   # everything except the last two items
# seq[:]                # [seq[0],   seq[1],          ..., seq[-1]    ]
# seq[low:]             # [seq[low], seq[low+1],      ..., seq[-1]    ]
# seq[:high]            # [seq[0],   seq[1],          ..., seq[high-1]]
# seq[low:high]         # [seq[low], seq[low+1],      ..., seq[high-1]]
# seq[::stride]         # [seq[0],   seq[stride],     ..., seq[-1]    ]
# seq[low::stride]      # [seq[low], seq[low+stride], ..., seq[-1]    ]
# seq[:high:stride]     # [seq[0],   seq[stride],     ..., seq[high-1]]
# seq[low:high:stride]  # [seq[low], seq[low+stride], ..., seq[high-1]]
#-----

# Step 3: Handling the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
# print(X[:, 1:3])
imputer.fit(X[:, 1:3])
# print(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X[:, 1:3])

# Step 4: Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# print(X)
# print(dataset)
# Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
# print(X)
# print(Y)

# Step 5: Splitting the datasets into training sets and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print(X_train)
# print(X_test)

# Step 6: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
# print(X_train)
# print(X_test)