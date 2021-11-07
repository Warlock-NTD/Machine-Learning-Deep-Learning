import numpy as np

def sigmoid(S):
    """
    :param S: an numpy array
    :return: sigmoid function of each element of S
    """
    return 1/ (1 + np.exp(-S))

def prob(w, X):
    """
    :param w: a 1d numpy array with d dimention
    :param X: a 2d numpy array with nxd dimention
    :return: sigmoid value to update estimate parameter
    """
    return sigmoid(X.dot(w))

def logistic_regression(w_init, X, y, lamb, learnrate = 0.1, nepoches = 2000):
    """
    :param w_init: weight init by default
    :param X: matrix data input
    :param y: matrix of label
    :param lamb: weight decay
    :param learnrate: constantly 0.1
    :param nepoches: constantly 2000 epoches
    :return: estimated param w (weight)
    """
    N, d = X.shape[0], X.shape[1]
    w = w_old = w_init
    ep = 0
    while ep < nepoches:
        ep += 1
        stochastic_ids = np.random.permutation(N)
        for i in stochastic_ids:
            xi = X[i]
            yi = y[i]
            # update estimate weight
            w = w - learnrate*((sigmoid(xi.T.dot(w)) - yi)*xi + lamb*w)
            if np.linalg.norm(w - w_old) < 1e-6:
                break
            w_old = w
    return w

np.random.seed(2)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# concatenate bias to Xbar
X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
w_init = np.random.randn(X.shape[1])
lamb = 0.0001
w = logistic_regression(w_init, X, y, lamb, learnrate=0.05, nepoches=500)
print('Solution of Logistic Regression:', w)