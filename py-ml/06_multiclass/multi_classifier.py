from re import I
import numpy as np
import mnist as mdata
# Just pass mdata to the train() and test()

# classify() updated to return matrix of 1 column 
# train() updated to 10 weights

# See page 67
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# See page 68, foward propagation is to predict
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

# Classify is a form of prediction, in this case, to 0 to 9, using the weightage.
def classify(X, w):
    y_hat = forward(X, w) # computes a matrix of predictions, 1 row per label, 1 column per class
    labels = np.argmax(y_hat, axis=1) # get index of max value in y_hat row, axis=1 means apply to each row
    return labels.reshape(-1, 1) # matrix of 1 column of labels

# See page 70 log loss formula 
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)

# The goal of gradient descent is to move downhill
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def report(iteration, X_train, Y_train, X_test, Y_test, w):
    matches = np.count_nonzero(classify(X_test, w) == Y_test)
    n_test_examples = Y_test.shape[0]
    matches = matches * 100.0 / n_test_examples
    training_loss = loss(X_train, Y_train, w)
    print("%d - Loss: %.20f, %.2f%%" % (iteration, training_loss, matches))

# Returns 10 weights after training
def train(X_train, Y_train, X_test, Y_test, iterations, lr):
    w = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for i in range(iterations):
        report(i, X_train, Y_train, X_test, Y_test, w)
        w -= gradient(X_train, Y_train, w) * lr
    report(iterations, X_train, Y_train, X_test, Y_test, w)
    return w

# Prepare data and train
w = train(mdata.X_train, mdata.Y_train, mdata.X_test, mdata.Y_test, iterations=200, lr=1e-5) #lr 0.00001
# Result
#200 - Loss: 0.08586319648804129068, 90.32%