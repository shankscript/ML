import pandas as pd
import quandl, math, datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = "bTUTXtVgF1K8yYL8uL-o"
df = quandl.get("BSE/BOM539551", authtoken="bTUTXtVgF1K8yYL8uL-o")
df = df[['Open', 'High', 'Low', 'Close', 'Total Turnover']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'Change', 'Total Turnover']]

label = 'Close'
df.fillna(-99999, inplace=True)

n = int(math.ceil(0.05 * len(df)))

df['label'] = df[label].shift(-n)

X = np.array(df.drop(['label'], 1))

X = preprocessing.scale(X)
X = X[:-n]
X_last = X[-n:]
df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

clf = LinearRegression(n_jobs=-1) #svm.SVR()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

predicts = clf.predict(X_last)

print(predicts, n, accuracy)

#print(X)
#print(df.tail())
#print(accuracy)
df['Forecast'] = np.nan

##
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in predicts:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
print(df.tail())
plt.show()
