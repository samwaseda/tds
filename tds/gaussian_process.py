import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.cluster import DBSCAN


class GaussianProcess:
    def __init__(self, function, min_samples=2, max_error=1.0e-3):
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
        if not self.is_enough:
            return np.inf
        return self.sigma_to_error * self.get_sigma(x)

    @property
    def sigma_to_error(self):
        if self._sigma_to_error is None:
            if len(self._sigma_lst) > 0:
                self._sigma_to_error = np.dot(
                    self._sigma_lst, self._error_lst
                ) / np.square(self._sigma_lst).sum()
            else:
                self._sigma_to_error = np.inf
        return self._sigma_to_error

    def extend(self, x):
        labels = DBSCAN(eps=0.001, min_samples=1).fit_predict(x)
        x = x[np.unique(labels, return_index=True)[1]]
        self._y_lst.extend(len(x) * [self._y_lst[-1]])
        self._x_lst.extend(x)

    def append(self, x):
        y_est = None
        if self.is_enough:
            self._sigma_lst.append(self.get_sigma(x))
            y_est = self.predict(x)
        self._x_lst.append(x)
        self._regressor = None
        y_real = np.squeeze(self.function(x))
        self._y_lst.append(y_real)
        if y_est is not None:
            self._error_lst.append(np.squeeze(np.absolute(y_est - y_real)))
            self._sigma_to_error = None

    @property
    def is_enough(self):
        return len(np.unique(self.y_lst)) > self.min_samples

    def get_arg_max_error(self, x):
        if not self.is_enough:
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
        return self.predict(x).reshape(np.asarray(x_in).shape[:-1])

    def predict(self, x_in):
        return np.squeeze(self.regressor.predict(np.atleast_2d(x_in)))

    @property
    def length(self):
        return self.regressor.kernel_.get_params()['k2__length_scale']

    def _get_x_diff(self, x):
        return (
            x.reshape(-1, x.shape[-1])[np.newaxis, :, :] - self.x_lst[:, np.newaxis, :]
        ) / self.length**2

    def _get_k_val(self, x):
        return self.regressor.kernel_(self.x_lst, x.reshape(-1, x.shape[-1]), eval_gradient=False)

    def get_gradient(self, x_in):
        x = np.array(x_in)
        return np.einsum(
            'i,ij,ijk->jk', self.regressor.alpha_, self._get_k_val(x), self._get_x_diff(x),
            optimize=True
        ).reshape(x.shape)

    def get_hessian(self, x_in):
        x = np.array(x_in)
        x_diff = self._get_x_diff(x)
        xx_diff = np.einsum('...i,...j->...ij', x_diff, x_diff) - 1 / self.length**2
        return np.einsum(
            'i,ij,ijkl->jkl', self.regressor.alpha_, self._get_k_val(x), xx_diff,
            optimize=True
        ).reshape(x.shape + (x.shape[-1], ))
