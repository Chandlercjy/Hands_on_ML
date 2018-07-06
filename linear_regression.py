import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LinearRegression:

    def __init__(self):
        self.cost_series = []

    def loss_func(self, x, y, theta):
        loss = x @ theta - y

        return loss

    def cost_func(self, loss):
        return np.sum(loss ** 2)/len(loss)

    def gradient_decent(self, x, y, theta, alpha, numIterations):

        for i in range(numIterations):
            loss = self.loss_func(x, y, theta)
            gradient = (x.T@loss) / len(y)  # 计算梯度
            theta = theta - alpha * gradient  # 参数theta的计算，即更新法则
            cost = self.cost_func(loss)
            self.cost_series.append(cost)

        return theta

    def plot_cost(self):
        plt.plot(self.cost_series)
        plt.show()


if __name__ == "__main__":

    df = pd.read_csv('data/linear_regression.csv')
    x1 = df.x1.values
    x2 = df.x2.values

    ones = np.ones(len(x1))

    # x1 = (x1-np.mean(x1))/np.std(x1)  # 归一化
    # x2 = (x2-np.mean(x2))/np.std(x2)  # 归一化

    # train_x = np.array([x1, ones]).T
    # train_y = df.y.values.reshape(len(df.y.values), 1)
    # train_w = np.array([[1], [1]])

    train_w = np.array([[1], [1], [1]])
    train_x = np.array([x1, x2, ones]).T
    train_y = df.y.values.reshape(len(df.y.values), 1)

    test = LinearRegression()
    test.gradient_decent(train_x, train_y, train_w, 0.01, 10000)
    test.plot_cost()
