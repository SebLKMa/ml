import numpy as np

# Predicts pizzas(y) from reservations(x)
# y = x * w, where w is the weight
def predict(X, w):
    return X * w

# loss refers to the error in prediction
# you get this error by comparing the actual X, Y from training data file vs result of predict(training data X, w)
# hence error = predict(X, w) -Y, which can result in -ve
# to avoid +ve loss, we square the prediction, error ** 2
def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

# lr refers to learning rate
# improve the weight until the loss is minimized
def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w
        
    raise Exception("Could not converge within %d iterations" % iterations)

# Load training data file
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system, i.e. get the line for prediction
w = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f" %w)

# Predict pizzas when reservations is 20
print("Prediction: x=%d => y%.2f" % (20, predict(20, w)))


