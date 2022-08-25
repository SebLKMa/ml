from re import I
import numpy as np
import mnist as mdata
# Just pass mdata to the train() and test()

# See page 67
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# See page 68, foward propagation is to predict
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

# Note that the labels used to train the classifier are either 0 or 1.
# Classify is a form of prediction, in this case, to either 0 or 1, using the weightage.
def classify(X, w):
    return np.round(forward(X, w))

# See page 70 log loss formula 
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)

# The goal of gradient descent is to move downhill
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

# Returns the weightage after training
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w

# Test the classification of inputs with weightage against the labels
def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results * 100 / total_examples
    print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))

# Prepare data and train
#x1, x2, x3, y = np.loadtxt("police.txt", skiprows=1, unpack=True)
#X = np.column_stack((np.ones(x1.size), x1, x2, x3))
#Y = y.reshape(-1, 1)
#w = train(X, Y, iterations=10000, lr=0.001)
w = train(mdata.X_train, mdata.Y_train, iterations=100, lr=1e-5) #lr 0.00001

# Test the classification with the "trained" weightage
#test(X, Y, w)
test(mdata.X_test, mdata.Y_test, w)