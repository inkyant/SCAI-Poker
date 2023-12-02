import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

from query import query 

# get data
csvData = pd.read_csv("Poker_data.csv") 

data_size = 1500

X = csvData.iloc[list(range(data_size//2)) + list(range(-data_size//2, 0)), 0:14].to_numpy()
y = csvData.iloc[list(range(data_size//2)) + list(range(-data_size//2, 0)), 14].to_numpy()

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

yPred = reg.predict(X_test)


print("Coefficients: \n", reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, yPred))
print("Coefficient of determination: %.2f" % r2_score(y_test, yPred))

def pred(cards):
    return reg.predict(np.array(cards).reshape(1, -1))

query(pred)