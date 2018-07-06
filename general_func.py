def loss_func(x, y, theta):
    loss = x @ theta - y

def sigmod(z):
    return 1/(1+np.exp(-z))
