
import datetime
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
import sklearn
from matplotlib import style
from sklearn import datasets, model_selection, preprocessing, svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import OnePy as op

style.use('ggplot')
# df = quandl.get('WIKI/GOOGL')
# df.to_csv('GOOGL.csv')
df = pd.read_csv('GOOGL.csv', index_col=0, parse_dates=True)
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change',  'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))  # 10% percent of data
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))

y = np.array(df['label'])
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2)

# clf = LinearRegression()
# clf.fit(X_train, y_train)
# with open('linear_regression.pkl', 'wb') as f:
# pickle.dump(clf, f)

pickle_in = open('linear_regression.pkl', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print(accuracy)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 60*60*24
next_unix = last_unix+one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Forecast'].plot()
# df = pd.read_csv('GOOGL.csv', index_col=0, parse_dates=True)
df['Adj. Close'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
