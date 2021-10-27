import numpy as np


def euclidean_distance(x, y):
    return np.sqrt((np.add.outer((x*x).sum(axis=-1), (y*y).sum(axis=-1)) - 2*np.dot(x, y.T)))

def cosine_distance(x, y):
    return 1.0 - np.dot(x, y.T)/np.sqrt(np.multiply.outer((x*x).sum(axis=-1), (y*y).sum(axis=-1)))
