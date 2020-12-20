import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp
from scipy.sparse.csr import csr_matrix

class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = False

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
    
        """
        w_ = w.copy() 
        if type(X) == csr_matrix:
            a = -csr_matrix.dot(X, w_).T*y
        else:   
            a = -np.matmul(X, w_).T*y
        z = np.zeros(a.shape[0])
        b = np.append(a,z)
        b = b.reshape(2, a.shape[0])
        w_[0] = 0
        return np.sum(logsumexp(b, axis=0))/X.shape[0] + self.l2_coef*np.dot(w_,w_)
        

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        w_ = w.copy()
        
        if type(X) == csr_matrix:
            a = -csr_matrix.dot(X, w_).T*y
            res = csr_matrix.dot(X.transpose(), expit(a)*y)
        else:   
            a = -np.matmul(X, w_).T*y
            res = np.dot(X.T, expit(a)*y)

        w_[0] = 0
        return -res/X.shape[0] + 2*self.l2_coef*w_
        

class MultinomialLoss(BaseLoss):
    """
    Loss function for multinomial regression.
    It should support l2 regularization.
    
    w should be 2d numpy.ndarray.
    First dimension is class amount.
    Second dimesion is feature space dimension.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = True

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 2d numpy.ndarray

        Returns
        -------
        : float
        """
        pass

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 2d numpy.ndarray

        Returns
        -------
        : 2d numpy.ndarray
        """
        pass

