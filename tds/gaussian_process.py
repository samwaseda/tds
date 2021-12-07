import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class GaussianProcess:
    def __init__(self, function, min_samples=5, max_error=1.0e-3):
        self.function = function
        self.min_samples = min_samples
        self.max_error = max_error
        self._x_lst = []
        self._y_lst = []
        self._sigma_lst = []
        self._error_lst = []
        self._sigma_to_error = None
        self._regressor = None

    @property
    def regressor(self):
        if self._regressor is None:
            kernel = ConstantKernel() * RBF()
            self._regressor = GaussianProcessRegressor(kernel=kernel)
            self._regressor.fit(self.x_lst, self.y_lst)
        return self._regressor

    @property
    def x_lst(self):
        return np.asarray(self._x_lst)

    @property
    def y_lst(self):
        return np.asarray(self._y_lst)

    def get_sigma(self, x):
        return np.squeeze(self.regressor.predict(np.atleast_2d(x), return_std=True)[1])

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

    def append(self, x, y_real=None):
        y_est = None
        if len(self.x_lst) > self.min_samples - 2:
            self._sigma_lst.append(self.get_sigma(x))
            y_est = self._get_value(x)
        self._x_lst.append(x)
        self._regressor = None
        if y_real is None:
            y_real = np.squeeze(self.function(x))
        self._y_lst.append(y_real)
        if y_est is not None:
            self._error_lst.append(np.squeeze(np.absolute(y_est - y_real)))
            self._sigma_to_error = None

    def get_arg_max_error(self, x):
        if len(self.x_lst) < self.min_samples:
            return np.random.permutation(x)[0]
        error = self.get_error(x)
        if np.max(error) < self.max_error:
            return None
        return x[np.argmax(error)]

    def get_value(self, x_in):
        x = np.asarray(x_in).reshape(-1, np.shape(x_in)[-1])
        while True:
            xx = self.get_arg_max_error(x)
            if xx is None:
                break
            self.append(xx)
        return self._get_value(x).reshape(np.asarray(x_in).shape[:-1])

    def _get_value(self, x_in):
        return np.squeeze(self.regressor.predict(np.atleast_2d(x_in)))
