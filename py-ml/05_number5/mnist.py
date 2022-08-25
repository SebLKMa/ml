import numpy as np
import gzip
import struct

# Training and Test data are download from http://yann.lecun.com/exdb/mnist/
# Refer to the url for data format details.

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

def encode_number5(Y):
    # Convert all 5s to 1s, others into 0s
    return (Y == 5).astype(int)

# Not all number images give the same Success or Accuracy
#for 5, Success: 9637/10000 (96.37%)
#for 1, Success: 9903/10000 (99.03%)
#for 8, Success: 9385/10000 (93.85%)

# (60000, 1) 60000 labels, each with value 1 if digit is a 5, and value 0 otherwise
Y_train = encode_number5(load_labels("train-labels-idx1-ubyte.gz"))
# (10000, 1) 10000 labels, with the same encoding as Y_train
Y_test = encode_number5(load_labels("t10k-labels-idx1-ubyte.gz"))


# 60000 images, 785 elements each (1 bias + 28 * 28 pixels)
# load_images() returns a matrix of (60000, 784) then prepends to (60000, 785)
X_train = prepend_bias(load_images("train-images-idx3-ubyte.gz"))
# 10000 images, 785 elememts each, same structure as X_train
# load_images() returns a matrix of (10000, 784) then prepends to (10000, 785)
X_test = prepend_bias(load_images("t10k-images-idx3-ubyte.gz"))