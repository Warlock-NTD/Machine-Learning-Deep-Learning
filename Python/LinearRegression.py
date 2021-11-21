from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]) .T
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# building Xbar
one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1).T

# calculate
psue_inv = np.linalg.pinv(np.dot(X, X.T))
tmp = np.dot(psue_inv, X)
res = np.dot(tmp, y)
w0, w = res[0], res[1]
y1 = w * 155 + w0
y2 = w * 160 + w0
print('Input 155cm, true output 52kg, predicted output %.2fkg.' %(y1) )
print('Input 160cm, true output 56kg, predicted output %.2fkg.' %(y2) )