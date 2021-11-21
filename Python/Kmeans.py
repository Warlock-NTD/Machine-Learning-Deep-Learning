from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random

np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis = 0)
K = 3   # number of clustering
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

def kmeans_init_centroids(X, K):
    # randomly pick k rows of X as initial centroids
    return X[np.random.choice(X.shape[0], K, replace=False)]

def kmeans_assign_labels(X, centroids):
    # calculate distance between data and current centroids
    D = cdist(X, centroids)
    # return index of closest centroid in 3 centroids.
    return np.argmin(D, axis=1)

def has_converged(centroids, new_centroids):
    # check and compare while current centroids and previous centroids are equal.
    return (set([tuple(a) for a in centroids]) == set(tuple(a) for a in new_centroids))

def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        # re assign label for all points
        Xk = X[labels == k, :]
        # take the value of normial.
        centroids[k, :] = np.mean(Xk, axis=0)
    return centroids

def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it += 1
    return centroids, labels, it


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    plt.plot(X0[:, 0], X0[:, 1], 'bs', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'gs', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)
    plt.axis('equal')
    plt.plot()
    plt.show()

centroids, labels, it = kmeans(X, K)
# print('Center :', centroids[-1])
kmeans_display(X, labels[-1])
# kmeans_display(X, original_label)
"""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)
"""