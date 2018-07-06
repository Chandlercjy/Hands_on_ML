import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Perceptron:

    def __init__(self, alpha, data, num_iter=100):
        self.w = np.zeros(2)
        self.b = 0
        self.data = data
        self.x = data[:, :2]
        self.y = data[:, 2]
        self.num_iter = num_iter

        self.alpha = alpha
        self.cost_series = []

    def fit(self):
        for i in range(self.num_iter):
            J = 0
            random.shuffle(self.data)  # 随机梯度下降

            for row in self.data:
                if self.is_error(row):
                    self.update_w(row)
                    self.update_b(row)
                    J += 1
                    print(f'更新第{i}次，w:{self.w},b:{self.b}')

            self.cost_series.append(J)

        return np.array(self.w, self.b)

    def is_error(self, row):
        value = row[-1]*(np.dot(row[:2], self.w)+self.b)

        return value <= 0

    def update_w(self, row):
        x = row[:2]
        y = row[-1]
        self.w += x*y*self.alpha

    def update_b(self, row):
        y = row[-1]
        self.b += y*self.alpha

    def plot_cost(self):
        plt.plot(self.cost_series)
        plt.show()


if __name__ == '__main__':
    DATA = np.array([[3, 3, 1], [4, 3, 1], [1, 1, -1]])
    test = Perceptron(1, DATA)
    answer = test.fit()
    test.plot_cost()

    # plt.scatter(DATA[:, 0], DATA[:, 1], c=DATA[:, 2], edgecolors='k')
    # plt.show()
