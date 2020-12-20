def encode_rle(x):
    import numpy as np
    n = len(x)
    y = np.array(x[1:] != x[:-1])
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return (x[i], z)
