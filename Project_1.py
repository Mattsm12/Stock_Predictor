import warnings
warnings.filterwarnings(action="ignore", module="sklearn", message="^internal gelsd")

import quandl
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#Write WIKI/input stock name
stock=input()
df = quandl.get(stock)

df = df[['Adj. Close']]

forecast_out = int(30)
#Prediction has to be based of Adjusted Closing Price and scales the points so there is no bias
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]
#Training data and tetsing data and fitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)
clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)
#Predicted Stock graph and the total change from beginning stock to ending stock and the average stock of the 30 days
starting = forecast_prediction[0]
ending = forecast_prediction[-1]

avg = sum(forecast_prediction) / float(len(forecast_prediction))

print("Total change in 30 days is : ", ending-starting)
print('The Average Stock is:', avg)

x = [0,29]
y = [starting,ending]
plt.plot(x,y)
plt.plot(forecast_prediction)
plt.show()
