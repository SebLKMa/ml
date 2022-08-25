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

# lr refers to learning rate
# improve the weight until the loss is minimized
# and improving the bias along the way
def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b
        
    raise Exception("Could not converge within %d iterations" % iterations)

# Load training data file
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system, i.e. get the line for prediction
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nweight=%.3f, bias=%.3f" % (w, b))

# Predict pizzas when reservations is 20
print("Prediction: x=%d => y%.2f" % (20, predict(20, w, b)))


