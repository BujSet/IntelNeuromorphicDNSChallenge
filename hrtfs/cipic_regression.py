from cipic_db import CipicDatabase
import numpy as np
from sklearn.linear_model import LinearRegression

CipicDatabase.subjects[3].collateAnthroData()

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
print(X)
print(y)
reg = LinearRegression().fit(X, y)
print(reg.score(X, y))