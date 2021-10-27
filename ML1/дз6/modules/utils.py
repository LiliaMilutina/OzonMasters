import numpy as np


def get_numeric_grad(f, x, eps):
    """
        Function to calculate numeric gradient of f function in x.
        
        Parameters
        ----------
        f : callable
        x : numpy.ndarray
        1d array, function argument
        eps : float
        Tolerance
        
        Returns
        -------
        : numpy.ndarray
        Numeric gradient.
        """
    
    e = np.eye(x.shape[0])
    
    x_new = x + e*eps
    
    x_ = np.tile(x, e.shape[0]).reshape(e.shape[0], e.shape[0])
    
    f_1 = np.apply_along_axis(f, 1, x_new)
    f_2 = np.apply_along_axis(f, 1, x_)
    return (f_1-f_2)/eps


def compute_balanced_accuracy(true_y, pred_y):
    """
        Get balaced accuracy value
        
        Parameters
        ----------
        true_y : numpy.ndarray
        True target.
        pred_y : numpy.ndarray
        Predictions.
        Returns
        -------
        : float
        """
    possible_y = set(true_y)
    value = 0
    for current_y in possible_y:
        mask = true_y == current_y
        value += (pred_y[mask] == current_y).sum() / mask.sum()
    return value / len(possible_y)


