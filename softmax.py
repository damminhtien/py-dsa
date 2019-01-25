import numpy as np
# z is numpy array


def softmax(z):
	x = np.exp(z)
	return x/np.sum(x)


z=np.array([1, 2, 3])
print('Soft max of [1, 2, 3]: ')
print(softmax(z))
