import numpy as np

# The mega-if statements in train() cannot scale with exponential combinations of cases.
# Gradient Descent is the faster, more precise and more general way to find the minimum loss.
def gradient(X, Y, w):
    return 2 * np.average(X * (predict(X, w, 0) - Y))

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

# lr refers to learning rate
# real world cannot afford us with exponential combination of IFs conditions.
# we need a scalable train(), using gradient descent
def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        print("Iteration %4d w: %.10f => Loss: %.10f" % (i, w, loss(X, Y, w, 0)))
        #g = gradient(X, Y, w)
        #print("Gradient: %.10f" % (g))
        #w -= g * lr
        w -= gradient(X, Y, w) * lr
    return w

# Load training data file
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system, i.e. get the line for prediction
w = train(X, Y, iterations=100, lr=0.001)
print("\nweight=%.10f" % w)

# Predict pizzas when reservations is 20
#print("Prediction: x=%d => y%.2f" % (20, predict(20, w, b)))


