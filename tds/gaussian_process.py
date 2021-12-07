import numpy as np


class GaussianProcess:
    def __init__(self, function, min_samples=4, max_error=1.0e-3, length=2):
        self.function = function
        self.min_samples = min_samples
        self.max_error = max_error
        self._x_lst = []
        self._y_lst = []
        self.length = length
        self._sigma_lst = []
        self._error_lst = []
        self._k_ij_inv = None
        self._sigma_to_error = None

    @property
    def x_lst(self):
        return np.asarray(self._x_lst)

    @property
    def y_lst(self):
        return np.asarray(self._y_lst)

    @property
    def k_ij_inv(self):
        if self._k_ij_inv is None:
            self._k_ij_inv = np.linalg.inv(self.get_k(self.x_lst, self.x_lst))
        return self._k_ij_inv

    def get_sigma(self, x):
        k_si = self.get_k_si(x)
        K = self.get_k_ss(x) - np.einsum('ij,jk,lk->il', k_si, self.k_ij_inv, k_si, optimize=True)
        return np.sqrt(np.squeeze(np.absolute(K.diagonal())))

    def get_error(self, x):
        if len(self.x_lst) < self.min_samples:
            return np.inf
        return self.sigma_to_error * self.get_sigma(x)

    @property
    def sigma_to_error(self):
        if self._sigma_to_error is None:
            self._sigma_to_error = np.dot(
                self._sigma_lst, self._error_lst
            ) / np.square(self._sigma_lst).sum()
        return self._sigma_to_error

    @property
    def divisor(self):
        return 2 * self.length**2

    def get_k(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        dx = x[..., :, np.newaxis, :] - y[..., np.newaxis, :, :]
        return np.exp(-np.sum(dx**2, axis=-1) / self.divisor)

    def get_k_ss(self, x):
        return self.get_k(x, x)

    def get_k_si(self, x):
        return self.get_k(x, self.x_lst)

    def append(self, x):
        y_est = None
        if len(self.x_lst) > self.min_samples - 2:
            self._sigma_lst.append(self.get_sigma(x))
            y_est = self._get_value(x)
        self._x_lst.append(x)
        self._k_ij_inv = None
        y_real = np.squeeze(self.function(x))
        self._y_lst.append(y_real)
        if y_est is not None:
            self._error_lst.append(np.squeeze(np.absolute(y_est - y_real)))
            self._sigma_to_error = None

    def get_value(self, x_in):
        n_dim = np.shape(x_in)[-1]
        for ii, xx in enumerate(np.random.permutation(np.asarray(x_in).reshape(-1, n_dim))):
            if self.get_error(xx) > self.max_error:
                self.append(xx)
        return self._get_value(x_in)

    def _get_value(self, x_in):
        return np.einsum('...i,ij,j->...', self.get_k_si(x_in), self.k_ij_inv, self.y_lst).squeeze()
