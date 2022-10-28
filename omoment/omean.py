import typing
import numbers
import numpy as np
import pandas as pd
from omoment import OBase


class OMean(OBase):
    __slots__ = ['mean', 'weight']

    def __init__(self, mean=np.nan, weight=None):
        mean = self._unwrap_if_possible(mean)
        weight = self._unwrap_if_possible(weight)

        if isinstance(mean, np.ndarray):
            mean, weight = self._mean_weight_of_np(mean, weight)

        self.mean = mean
        self.weight = weight or 0
        self._validate()

    def _validate(self):
        if self.weight < 0:
            raise ValueError('Weight cannot be negative.')
        elif np.isnan(self.weight) | np.isinf(self.weight):
            raise ValueError('Invalid weight provided.')
        elif self.weight > 0 and (np.isnan(self.mean) or np.isinf(self.mean)):
            raise ValueError('Invalid mean provided.')

    @staticmethod
    def _mean_weight_of_np(x, w=None, raise_if_nans=False):
        # check if we have np.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError(f'x has to be a np.ndarray, it is {type(x)}.')
        # more than 1-dimensional array, throw an exception
        elif x.ndim > 1:
            raise ValueError(f'Provided np.ndarray has to be 1-dimensional (x.ndim = {x.ndim}).')
        # 1-dimensional array (0-dimensional should not be there, but it should not matter anyway
        else:
            if w is None:
                invalid = np.isnan(x) | np.isinf(x)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                return np.mean(x[~invalid]), np.sum(~invalid)
            else:
                if w.ndim > 1 or len(w) != len(x):
                    raise ValueError('w has to have the same shape and size as x')
                invalid = np.isnan(x) | np.isinf(x) | np.isnan(w) | np.isinf(w)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = np.sum(w[~invalid])
                mean = np.average(x[~invalid], weights=w[~invalid])
                return mean, weight

    def update(self, x, w=None, raise_if_nans=False):
        """
        Update the moments by adding some values; NaNs are removed both from values and from weights.

        Args:
            x (Union[float, np.ndarray, pd.Series]): Values to add to the estimator.
            w (Union[float, np.ndarray, pd.Series]): Weights for new values. If provided, has to have the same length
            as x.
        """
        x = self._unwrap_if_possible(x)
        w = self._unwrap_if_possible(w)
        if isinstance(x, np.ndarray):
            x, w = self._mean_weight_of_np(x, w)
        self += OMean(x, w)
        return self

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            return OMean(other.mean, other.weight)
        elif other.weight == 0:
            return OMean(self.mean, self.weight)
        else:
            delta = other.mean - self.mean
            new_weight = self.weight + other.weight
            new_mean = self.mean + delta * other.weight / new_weight
            return OMean(mean=new_mean, weight=new_weight)

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            self.mean = other.mean
            self.weight = other.weight
            return self
        elif other.weight == 0:
            return self
        else:
            delta = other.mean - self.mean
            self.weight += other.weight
            self.mean += delta * other.weight / self.weight
            return self

    def __nonzero__(self):
        return self.weight.__nonzero__()

    @classmethod
    def of_frame(cls, data, x, w=None):
        res = cls()
        if w is None:
            res.update(data[x].values)
            return res
        else:
            res.update(data[x].values, w=data[w].values)
            return res
