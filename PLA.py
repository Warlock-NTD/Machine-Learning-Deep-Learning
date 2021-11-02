from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# generate data
# list of data point
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3 , .2], [.2, .3]]

N = 10

x0 = np.random.multivariate_normal(means[0], cov, N).T
x1 = np.random.multivariate_normal(means[1], cov, N).T

x = np.concatenate((x0, x1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis = 1)

X = np.concatenate((np.ones((1, 2*N)), x), axis = 0)

cov1 = np.cov(np.array([[2, 2], [8, 3], [3, 6]]).T)
print(cov1)

