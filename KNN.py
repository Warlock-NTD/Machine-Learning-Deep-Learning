from __future__ import print_function
import numpy as np
from time import time

d, N = 1000, 10000

X = np.random.randn(N, d)
z = np.random.randn(d)

# square distance between 2 vector z and x.
def dist_pp(z, x):
    d = z - x.reshape(z.shape)
    return np.sum(d*d)


# distance from 1 point to each point in a set (Solution 1).
def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range (N):
        res[0][i] = dist_pp(z, X[i])
    return res

# distance from 1 point to each point in a set (Solution 2).
def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1)
    z2 = np.sum(z*z)
    return X2 + z2 - 2*X.dot(z)

# computing
"""
t1 = time()
D1 = dist_ps_naive(z, X)
print('naive running time :', time() - t1)

t1 = time()
D2 = dist_ps_fast(z, X)
print('fast running time :', time() - t1)
print('diff result naive vs fast:', np.linalg.norm(D1 - D2))
"""

Z = np.random.randn(100, d)

# distance from each point in Set Z to each point in Set X (Solution 1).
def dist_ss_0(Z, X):
    M, N = Z.shape[0], X.shape[0]
    res = np.zeros((M, N))
    for i in range (M):
        res[i] = dist_ps_fast(Z[i], X)
    return res

# distance from each point in Set Z to each point in Set X (Solution 2).
def dist_ss_fast(Z, X):
    X2 = np.sum(X*X, 1)
    Z2 = np.sum(Z*Z, 1)
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)

t1 = time()
D3 = dist_ss_0(Z, X)
print('semi fast set to set running time:', time() - t1)
t1 = time()
D4 = dist_ss_fast(Z, X)
print('fast set to set running time:', time() - t1)
print('result diff:', np.linalg.norm(D3 - D4))