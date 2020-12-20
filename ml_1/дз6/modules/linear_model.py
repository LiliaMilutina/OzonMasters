import numpy as np
from scipy.special import expit
import time
from scipy.sparse.csr import csr_matrix
from sklearn.metrics import balanced_accuracy_score

class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
            
        if w_0 is None:
            w_0 = np.random.rand(X.shape[1])
        
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        
        time_ = []
        func_ = []
        func_val_ = []
        ac_ = []

        num_iter = int(X.shape[0]/self.batch_size)
        
        ac = 1

        for k in range (1, self.max_iter+1):
            np.random.seed(self.random_seed)
            rand_perm = np.random.permutation(np.arange(0, X.shape[0]))
            
            start_time = time.time()
            
            eta = self.step_alpha/np.power(k, self.step_beta)

            st = 0
            fi = self.batch_size
            for i in range(num_iter):
                ind = rand_perm[st:fi]
                X_train = X[ind, :]
                y_train = y[ind]

                temp = self.loss_function.grad(X_train, y_train, w_0)
                w_new = w_0 - eta*temp
                
                ac = np.dot(w_new-w_0, w_new-w_0)

                w_0 = w_new
                st += self.batch_size
                
                if (fi + self.batch_size > X.shape[0]):
                    break
                else:
                    fi += self.batch_size
                    
                if ac <= self.tolerance:
                    break
            
            self.w = w_new
            if trace is True:
                func_.append(self.loss_function.func(X, y, w_0))
                if X_val is None:
                    func_val_.append(None)
                else:
                    func_val_.append(self.loss_function.func(X_val, y_val, w_0))    
                    
                    if self.batch_size != X.shape[0]:
                        threshold = self.get_optimal_threshold(X_val, y_val)
                        y_pred = self.predict(X_val, threshold)
                        ac_.append(balanced_accuracy_score(y_val, y_pred))
                time_.append(time.time() - start_time)
            if ac <= self.tolerance:
                break
                    
        if trace is True:
            history = dict()
            history['func'] = func_
            history['func_val'] = func_val_
            history['time'] = time_
            history['accuracy'] = ac_
            return history
        
            
    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
            
        """
        if type(X) == csr_matrix:
            y_pr = csr_matrix.dot(X, self.w)
        else:
            y_pr = np.matmul(self.w, X.T)
        
        return np.where(y_pr>threshold, 1, -1)

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """
        if self.loss_function.is_multiclass_task:
            raise TypeError('optimal threhold procedure is only for binary task')

        weights = self.get_weights()
        scores = X.dot(weights)
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        
        return self.loss_function.func(X, y, self.w)