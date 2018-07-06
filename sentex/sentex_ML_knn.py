import random
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import model_selection, neighbors, preprocessing

style.use('fivethirtyeight')


def sklearn_knn():
    df = pd.read_csv('classify_data.csv')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()

    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])

    prediction = clf.predict(example_measures)
    print(prediction)


def calculate_euclidean_distance(x: np.array, y: np.array) -> np.array:
    return np.linalg.norm(x-y)


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []

    for group in data:
        for features in data[group]:
            distance = calculate_euclidean_distance(features, predict)
            distances.append([distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k

    return vote_result


def test1():

    data = dict(k=np.array([[1, 2], [2, 3], [3, 1]]),
                r=np.array([[6, 5], [7, 7], [8, 6]]))
    new_features = np.array([[5, 7]])

    result = k_nearest_neighbors(data, new_features, 3)
    print(result)

    [plt.scatter(ii[0], ii[1]) for i in data for ii in data[i]]
    plt.scatter(new_features[0][0], new_features[0][1], s=100)

    plt.show()


df = pd.read_csv('knn_data.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values
random.shuffle(full_data)
