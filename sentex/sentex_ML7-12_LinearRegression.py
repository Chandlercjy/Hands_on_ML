import random

import matplotlib.pyplot as plt
import numpy as np


def create_data_sets(val_num,  variance=1, step=2, correlation=False):
    val = 1
    ys = []

    for i in range(val_num):
        y = val + random.randrange(-variance, variance)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
        ys.append(y)

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def calculate_slope(xs, ys):
    # return np.cov(xs, ys)[0][1]/np.var(xs)
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)**2)-np.mean(xs**2)))

    return m


def calculate_intercept(xs, ys):
    slope = calculate_slope(xs, ys)

    return np.mean(ys - xs*slope)


def calculate_R_square(ys, est_ys):
    return np.corrcoef(ys, est_ys)[0][1]**2


xs, ys = create_data_sets(40,  40, 2, 'pos')

slope = calculate_slope(xs, ys)
intercept = calculate_intercept(xs, ys)
regression_line = slope*xs + intercept

pred_x = 8
pred_y = slope*pred_x + intercept

r_square = calculate_R_square(ys, regression_line)
print('R square:', r_square)
print(f'Linear Regression: y = {slope:.2f}x + {intercept:.2f}')

# plt.scatter(xs, ys)
# plt.plot(regression_line)
# plt.scatter(pred_x, pred_y, linewidths=5, color='r')

# plt.show()
