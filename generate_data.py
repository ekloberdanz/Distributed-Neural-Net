import nnfs
from nnfs.datasets import spiral_data
import numpy as np

nnfs.init()
# Create dataset
# X, y = spiral_data(samples=1000, classes=3)

# split = int(X.shape[0] * 0.7)
# split
# X_train = X[:split]
# y_train = y[:split]
# X_test = X[split:]
# y_test = y[split:]


X_train, y_train = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

np.savetxt("./data/X_train.csv", X_train, delimiter=",")
np.savetxt("./data/y_train.csv", y_train, delimiter=",")
np.savetxt("./data/X_test.csv", X_test, delimiter=",")
np.savetxt("./data/y_test.csv", y_test, delimiter=",")