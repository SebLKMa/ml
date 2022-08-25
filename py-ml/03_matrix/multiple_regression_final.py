import numpy as np
from numpy.core.numeric import ones

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
        w -= gradient(X, Y, w) * lr
    return w

x1, x2, x3 , y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
# the trick add back the bias by just adding another x variable to formula and the matrix
#    y = x1 * w1 + x2 * w2 + x3 * w3 + x0 * b
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
# reshape 1 dimension array y to matrix Y with 1 column with as many rows as y
Y = y.reshape(-1, 1)
# start training
w = train(X, Y, iterations=100000, lr=0.001)

print("\nThe transposed Weights: [[ Bias, Reservations, Temperature, Tourists]]")
print("               Weights: %s" % w.T)
print("\nA few -> predictions compared to actual(label)")
for i in range(5):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))

# print() I/O slows Python
# The Go version is faster than the Python (start both at the same time to compare)