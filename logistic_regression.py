import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

def sigmod(z):
    return 1/(1+np.exp(-z))


def sign(value):
    value[value >= 0] = 1
    value[value < 0] = -1

    return value


class LogisticRegression:

    def __init__(self):
        self.cost_series = []

    def hypothesis(self, x, y, theta):
        return sigmod(x @ theta)

    def cost(self, h, y):

        return -(y.T @ np.log(h) + (1-y).T @ np.log(1-h))/len(y)

    def gradient_decent(self, x, y, theta, alpha, numIterations):

        for i in range(numIterations):
            h = self.hypothesis(x, y, theta)
            gradient = (x.T@(h-y)) / len(y)  # 计算梯度
            theta = theta - alpha * gradient  # 参数theta的计算，即更新法则
            cost = self.cost(h, y)
            self.cost_series.append(cost[0])

        return theta

    def plot_cost(self):
        plt.plot(self.cost_series)
        plt.show()


class Perceptron:

    def __init__(self):
        self.cost_series = []

    def loss(self, x, y, theta):
        return -x @ theta*y

    def cost(self, loss):
        return np.sum(loss)

    def random_gradient_decent(self, x, y, theta, alpha, num_iterations):

        for i in range(num_iterations):
            alpha = 5/(i+1)
            total = np.c_[x, y]
            random.shuffle(total)
            b = theta

            J = 0

            for row in total:
                gradient_w = row[-1]*row[1:-1]  # 对theta
                gradient_b = row[0]  # 对常数项

                if row[:-1] @ theta * row[-1] <= 0:
                    update_w = theta[1:-1].T+alpha * gradient_w
                    update_b = theta[0]+alpha * gradient_b
                    theta = theta-np.c_[update_b, update_w].T
                    J += 1
            cost = self.cost(self.loss(x, y, theta))
            self.cost_series.append(J)

        return theta

    def plot_cost(self):
        plt.plot(self.cost_series)
        plt.show()


if __name__ == "__main__":

    df = pd.read_csv('data/perceptron.csv')
    x1 = df.x1.values
    x2 = df.x2.values

    ones = np.ones(len(x1))

    X = np.array([x1, x2]).T

    x_max, x_min, x_mean = X.max(), X.min(), X.mean()
    X = (X-x_mean)/(x_max-x_min)

    X = np.c_[ones, X]

    Y = df.y.values.reshape(len(df.y.values), 1)
    Y[Y == 0] = -1
    T = np.array([[1], [1], [1]])

    test = Perceptron()
    a = test.random_gradient_decent(X, Y, T, 0.1, 100)
    # H = test.hypothesis(X, Y, T)
    # cost = test.cost(H, Y)
    test.plot_cost()

    # plt.scatter(X[:, 1], X[:, 2], c=Y.T[0], edgecolors='k')
    # plt.show()
