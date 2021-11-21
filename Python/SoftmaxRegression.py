import numpy as np

def softmax(Z):
    """
    :param Z: a numpy array of shape (N, C), C is number of classified
    :return: a numpy array of shape (N, C).
    """
    exZ = np.exp(Z)
    SM = exZ / exZ.sum(axis = 1, keepdims = True)
    return SM

def softmax_stable(Z):
    exZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    SM = exZ / np.sum(axis = 1, keepdims=True)
    return SM


