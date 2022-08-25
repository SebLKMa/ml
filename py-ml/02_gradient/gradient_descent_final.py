import time
import numpy as np

# Adding Bias can helper to lower the error/loss
# Predicts pizzas(y) from reservations(x)
# y = x * w + b, where is the weight and b is the bias
def predict(X, w, b):
    return X * w + b

# loss refers to the error in prediction
# you get this error by comparing the actual X, Y from training data file vs result of predict(training data X, w)
# hence error = predict(X, w) -Y, which can result in -ve
# to avoid +ve loss, we square the prediction, error ** 2
def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

# The mega-if statements in train() cannot scale with exponential combinations of cases.
# Gradient Descent is the faster, more precise and more general way to find the minimum loss.
# 3D plane Weight, Bias, Loss(x, y, z)
def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)

# lr refers to learning rate
# real world cannot afford us with exponential combination of IFs conditions.
# we need a scalable train(), using gradient descent
def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        #print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        wg, bg = gradient(X, Y, w, b)
        w -= wg * lr
        b -= bg * lr
    return w, b

# Load training data file
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system, i.e. get the gradients for prediction
# AttributeError: module 'time' has no attribute 'time_ns' <--- means need to upgrade to python3.7, and re-install numpy
# https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-7-on-ubuntu-18-10/
start = time.time_ns()
w, b = train(X, Y, iterations=20000, lr=0.001)
end = time.time_ns()
print(f"train() completed in: {end - start}ns") # the print() i/o in train() causes much delay

print("\nweight=%.10f bias=%.10f" % (w, b))

# Predict pizzas when reservations is 20
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

# Compare the result with the much more iterations and learning rate in
# linear_regression_bias_more_iters.py
