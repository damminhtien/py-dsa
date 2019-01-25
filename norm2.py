import numpy as np


def norm2(z):
	return z.dot(z)**0.5


z = np.array([1, 2, 3])
print(norm2(z))
