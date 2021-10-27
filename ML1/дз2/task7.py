def calc_expectations(h, w, X, Q):
    import numpy as np
    cum_1 = np.cumsum(Q, axis=0)
    cum_1[h:, :] = cum_1[h:, :] - cum_1[:-h, :]
    cum_2 = np.cumsum(cum_1, axis=1)
    cum_2[:, w:] = cum_2[:, w:] - cum_2[:, :-w]
    res = cum_2*X
    return res
