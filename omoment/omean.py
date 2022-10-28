import numpy as np
import pandas as pd
import typing
from omoment import OBase


class OMean(OBase):
    __slots__ = ['mean', 'weight']

    def __init__(self, mean=np.nan, weight=0):
        if isinstance(mean, float):
            self.mean = mean
            self.weight = weight
            self._validate()
        else:
            self.mean = np.nan
            self.weight = 0
            if weight == 0:
                weight = 1
            self.update(mean, weight)

    def _validate(self):
        if self.weight < 0:
            raise ValueError('Weight cannot be negative.')
        elif np.isnan(self.weight) | np.isinf(self.weight):
            raise ValueError('Invalid weight provided.')
        elif self.weight > 0 and (np.isnan(self.mean) or np.isinf(self.mean)):
            raise ValueError('Invalid mean provided.')

    def update(self, x, w=1, raise_if_nans=False):
        """
        Update the moments by adding some values; NaNs are removed both from values and from weights.

        Args:
            x (Union[float, np.ndarray, pd.Series]): Values to add to the estimator.
            w (Union[float, np.ndarray, pd.Series]): Weights for new values. If provided, has to have the same length
            as x.
        """
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(w, pd.Series):
            w = w.values
        if isinstance(x, np.ndarray):
            if x.size == 1:
                x = float(x)
                if isinstance(w, np.ndarray):
                    if w.size != 1:
                        raise ValueError(f'w (size={w.size}) has to have the same size as x (size={x.size})!')
                    w = float(w)
            elif x.ndim > 1:
                raise ValueError(f'Provided np.ndarray has to be 1-dimensional (x.ndim = {x.ndim}).')
            elif isinstance(w, np.ndarray):
                if w.ndim > 1 or len(w) != len(x):
                    raise ValueError('w has to have the same shape and size as x!')
                invalid = np.isnan(x) | np.isinf(x) | np.isnan(w) | np.isinf(w)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = np.sum(w[~invalid])
                mean = np.average(x[~invalid], w[~invalid])
                self += OMean(mean, weight)
                return
            else:
                if w != 1:
                    raise ValueError(f'If x is np.ndarray, w has to be too and has to have the same size.')
                invalid = np.isnan(x) | np.isinf(x)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = len(x[~invalid])
                mean = x[~invalid].mean()
                self += OMean(mean, weight)
                return
        self += OMean(x, w)

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
        elif other.weight == 0:
            pass
        else:
            delta = other.mean - self.mean
            self.weight += other.weight
            self.mean += delta * other.weight / self.weight

    def __nonzero__(self):
        return self.weight.__nonzero__()

    @classmethod
    def of_frame(cls, data, x, w=None):
        res = cls()
        if w is None:
            return res.update(data[x].values)
        else:
            return res.update(data[x].values, w=data[w].values)
