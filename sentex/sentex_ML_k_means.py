import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import preprocessing
from sklearn.cluster import KMeans

style.use('ggplot')


def download_data():
    df = pd.read_excel(
        'https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls')
    df.to_csv('titanic.csv', index=False)


def handle_non_numerical_data(data: pd.DataFrame):
    columns = data.columns.values.tolist()

    for column in columns:

        if data[column].dtypes not in [np.int64, np.float64]:
            unique_val = set(data[column].values.tolist())
            unique_dict = {value: index for index,
                           value in enumerate(unique_val)}
            data[column] = data[column].map(unique_dict)

    return data


def check_accuracy(x, y, clf):
    correct = 0

    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))

        prediction = clf.predict(predict_me)

        if prediction[0] == y[i]:
            correct += 1

    print(correct/len(X))


df = pd.read_csv('titanic.csv')
df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)
df = handle_non_numerical_data(df)
df.drop(['boat', 'sex'], 1, inplace=True)
print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

check_accuracy(X, y, clf)
