import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('deneyim-maas.csv', delimiter=";")

plt.scatter(dataset.deneyim, dataset.maas)
X = dataset.iloc[:, :1].values
y = dataset.iloc[:,1:].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state=1)

#plt.scatter(y_test, X_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, lr.predict(X_train), color="blue",)
plt.title("Train data set")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.show()