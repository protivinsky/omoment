import numpy as np
import pandas as pd
import typing
from omoment import OMean

class OMeanVar(OMean):
    __slots__ = ['mean', 'var', 'weight']

    def __init__(self, mean=np.nan, var=np.nan, weight=0):
        if isinstance(mean, float):
            self.mean = mean
            self.var = var
            self.weight = weight
            self._validate()
        else:
            self.mean = np.nan
            self.var = np.nan
            self.weight = 0
            if weight == 0:
                weight = 1
            self.update(mean, weight)

    def _validate(self):
        OMean._validate(self)
        if self.weight > 0 and (np.isnan(self.var) or np.isinf(self.var)):
            raise ValueError('Invalid variance provided.')

    def update(self, x, w=1, raise_if_nans=False):
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
                mean = np.average(x[~invalid], weights=w[~invalid])
                var = np.average((x[~invalid] - mean) ** 2, weights=w[~invalid])
                self += OMeanVar(mean, var, weight)
                return
            else:
                if w != 1:
                    raise ValueError(f'If x is np.ndarray, w has to be too and has to have the same size.')
                invalid = np.isnan(x) | np.isinf(x)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = len(x[~invalid])
                mean = np.mean([~invalid])
                var = np.mean((x[~invalid] - mean) ** 2)
                self += OMeanVar(mean, var, weight)
                return

        self += OMeanVar(x, 0, w)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            return OMeanVar(other.mean, other.var, other.weight)
        elif other.weight == 0:
            return OMeanVar(self.mean, self.var, self.weight)
        else:
            delta_mean = other.mean - self.mean
            delta_var = other.var - self.var
            new_weight = self.weight + other.weight
            ratio = other.weight / new_weight
            new_mean = self.mean + delta_mean * ratio
            new_var = self.var + delta_var * ratio + delta_mean ** 2 * ratio * (1 - ratio)
            return OMeanVar(new_mean, new_var, new_weight)

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            return OMeanVar(other.mean, other.var, other.weight)
        elif other.weight == 0:
            return OMeanVar(self.mean, self.var, self.weight)
        else:
            delta_mean = other.mean - self.mean
            delta_var = other.var - self.var
            ratio = other.weight / (self.weight + other.weight)
            self.mean = self.mean + delta_mean * ratio
            self.var = self.var + delta_var * ratio + delta_mean ** 2 * ratio * (1 - ratio)
            self.weight = self.weight + other.weight

    @property
    def std_dev(self):
        return np.sqrt(self.var)

    @property
    def unbiased_var(self):
        return self.var * (self.weight / (self.weight - 1))

    @property
    def unbiased_std_dev(self):
        return np.sqrt(self.unbiased_var)

