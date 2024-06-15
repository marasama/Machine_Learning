import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv", delimiter=";")
X = dataset.iloc[:, 0:3].values
y  = dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer as sim

imputer = sim(missing_values=np.nan, strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder as oHe

ct = ColumnTransformer(transformers=[("encoder", oHe(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train[:, 3:] = ss.fit_transform(X_train[:, 3:])
X_test[:, 3:] = ss.transform(X_test[:, 3:])

print(X)
print(y)