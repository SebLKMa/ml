import numpy as np
import gzip
import struct

# Training and Test data are download from http://yann.lecun.com/exdb/mnist/
# Refer to the url for data format details.

# encode_number5() is replaced by one_hot_encode()

def load_images(filename):
    # Open, unzip and decode binary file to images
    with gzip.open(filename, 'rb') as f:
        # Read header into a bunch of variables
        # ">IIII" indicates pattern to read 
        # - 4 unsigned Integers encoded with most significant byte first
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all pixels into NumPy array of bytes
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix, each line is an image
        return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
    # Insert a bias column of 1s in position 0 of X
    # "axis=1" means "insert a column, not a row"
    return np.insert(X, 0, 1, axis=1)

def load_labels(filename):
    # Open and unzip
    with gzip.open(filename, 'rb') as f:
        # Skip header bytes
        f.read(8)
        # Read all labels into a list
        all_labels = f.read()
        # Reshape list into a 1-column matrix
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)

# Refer to second diagram on page 89
def one_hot_encode(Y):
    n_labels = Y.shape[0] # the no. of rows in Y
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes)) # initialize all to 0s. 1 row per label, 1 column per class.
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1 # flips hot value to 1
    return encoded_Y

# (60000, 1) 60000 labels, each a single digit 0 to 9
Y_train = one_hot_encode(load_labels("../05_number5/train-labels-idx1-ubyte.gz"))

# (10000, 1) 10000 labels, each a single digit 0 to 9,
# refer to the updated classify() as to why one_hot_encode() is not needed for Y_test
# we are going to compare Y_test with the classifier's output.
Y_test = load_labels("../05_number5/t10k-labels-idx1-ubyte.gz")


# 60000 images, 785 elements each (1 bias + 28 * 28 pixels)
# load_images() returns a matrix of (60000, 784) then prepends to (60000, 785)
X_train = prepend_bias(load_images("../05_number5/train-images-idx3-ubyte.gz"))
# 10000 images, 785 elememts each, same structure as X_train
# load_images() returns a matrix of (10000, 784) then prepends to (10000, 785)
X_test = prepend_bias(load_images("../05_number5/t10k-images-idx3-ubyte.gz"))