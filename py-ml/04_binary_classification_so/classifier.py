import numpy as np
import binary_classifier as bc # binary_classifier is from binary_classifier.so

# Prepare data and train
x1, x2, x3, y = np.loadtxt("police.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = bc.train(X, Y, iterations=10000, lr=0.001)
# This iterations and lr gives
#Success: 25/30 (83.33%)

print("\nThe transposed Weights: [[ Bias, Reservations, Temperature, Tourists]]")
print("               Weights: %s" % w.T)

# Test the classification with the "trained" weightage
bc.test(X, Y, w)

# Note that increasing the Iterations to 50000 for training increases the success percent as well
#Iteration 49999 => Loss: 0.32928644915909671687
#Success: 28/30 (93.33%)

# lr set to 1e-5 (0.00005)
#Iteration 9999 => Loss: 0.53535851717232207925
#Success: 26/30 (86.67%)