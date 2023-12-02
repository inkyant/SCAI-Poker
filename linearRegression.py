import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


reg = linear_model.LinearRegression()
x = np.append(np.random.randint(0, 6, (500, 13)), np.random.randint(7, 13, (500, 13)), axis=0)
y = np.append(np.zeros((500)), np.ones(500), axis=0)
reg.fit(x, y)

yPred = reg.predict(x)


print("Coefficients: \n", reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y, yPred))
print("Coefficient of determination: %.2f" % r2_score(y, yPred))




