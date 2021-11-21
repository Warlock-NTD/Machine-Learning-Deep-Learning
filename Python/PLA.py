from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def predict(w, X):
    """
    predict label of each row of X, given w
    X: a 2-d array of shape (N, d), each row is a datapoint --> 20x3
    w: a 1-d numpy array of shape (d) --> 3x1
    result --> vector 20x1.
    """
    return np.sign(X.dot(w))

def perceptron(X, y, w_init):
    """ perform perception learning algorithm
    X: a 2-d numpy array of shape (N, d), each row is datapoint
    y: a 1-d numpy array of shape (N), label of each row of X. y[i] = 1/ -1
    w_init: a 1-d numpy array of shape (d)
    """
    w = w_init
    while True:
        pred = predict(w, X)
        # find indexing of missclassified points
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        # compare numpy equal with 2 array pred(20x1) and y(20x1), equal == False --> non same sign
        # return numpy array mis_idxs of several index of elements that stisfy condition : pred # y
        num_miss = mis_idxs.shape[0]
        if num_miss == 0:
            return w
        # randomly pick one of missed point
        random_id = np.random.choice(mis_idxs, 1)[0]
        # return ndarray of random in size 1, similar to single item, but needed to perform [0] index
        # np.random.choice[0] has numpy.int64 type and np.random.choice has numpy.ndarray type
        # update weight var
        w = w + y[random_id] * X[random_id]
    return w

# generate data
# list of data point
# np.random.seed(2)

means = [[-1, 0], [1, 0]]
cov = [[.3 , .2], [.2, .3]]

N = 10

x0 = np.random.multivariate_normal(means[0], cov, N) # 10x2
x1 = np.random.multivariate_normal(means[1], cov, N) # 10x2
# each datapoint combine in each row, each data has 2 feature

x = np.concatenate((x0, x1), axis = 0) # 20x2

y = np.concatenate((np.ones((N)), -1 * np.ones((N)))) # label numpy array... 20x1

# building Xbar by giving feature x0 = 1 for bias w0
Xbar = np.concatenate((np.ones((2*N, 1)), x), axis = 1) # 20x3

"""
due to this step , we have built Xbar with 20 datapoints
each data point has 2 main feature and 1 extra feature by value default = 1.
sample a datapoint x1(1,a1,a2).
"""
w_init = np.random.randn(Xbar.shape[1])

# Xbar.shape[1] = 3, the dimention of datapoint similar to weight var w.
# estimation w. (1, w1, w2, w3, ... wn)
w = perceptron(Xbar, y, w_init)

