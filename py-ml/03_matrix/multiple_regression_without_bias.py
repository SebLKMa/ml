import numpy as np

def predict(X, w):
    return np.matmul(X, w)

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

def gradient(X, Y, w):
    # X.T means matrix X transposed
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        g = gradient(X, Y, w)
        print(g)
        w -= g * lr
        #w -= gradient(X, Y, w) * lr
    return w

x1, x2, x3 , y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
X = np.column_stack((x1, x2, x3))
# reshape 1 dimension array y to matrix Y with 1 column with as many rows as y
Y = y.reshape(-1, 1)
# start training
w = train(X, Y, iterations=10, lr=0.001)
