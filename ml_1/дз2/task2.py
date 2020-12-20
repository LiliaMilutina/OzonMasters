def replace_nan_to_means(X):
    import numpy as np
    ind_nan = np.where(np.isnan(X))
    nan_mean = np.nanmean(X, axis=0)
    X[ind_nan] = np.take(nan_mean, ind_nan[1])
    return X
