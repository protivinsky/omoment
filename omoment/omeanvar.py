from __future__ import annotations
from numbers import Number
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from omoment import OMean


class OMeanVar(OMean):
    __slots__ = ['mean', 'var', 'weight']

    def __init__(self,
                 mean: Union[Number, np.ndarray, pd.Series] = np.nan,
                 var: Optional[Number] = None,
                 weight: Union[Number, np.ndarray, pd.Series] = None) -> OMeanVar:
        mean = self._unwrap_if_possible(mean)
        weight = self._unwrap_if_possible(weight)

        if isinstance(mean, np.ndarray):
            if var is not None:
                raise ValueError('var cannot be provided if mean is a np.ndarray.')
            mean, var, weight = self._mean_var_weight_of_np(mean, weight)

        self.mean = mean
        self.var = var or np.nan
        self.weight = weight or 0
        self._validate()

    def _validate(self):
        OMean._validate(self)
        if (self.weight > 0 and (np.isnan(self.var) or np.isinf(self.var))) or self.var < 0:
            raise ValueError('Invalid variance provided.')

    @staticmethod
    def _mean_var_weight_of_np(x: np.ndarray,
                               w: Optional[np.ndarray] = None,
                               raise_if_nans: bool = False) -> Tuple[float, float, float]:
        # check if we have np.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError(f'x has to be a np.ndarray, it is {type(x)}.')
        # more than 1-dimensional array, throw an exception
        elif x.ndim > 1:
            raise ValueError(f'Provided np.ndarray has to be 1-dimensional (x.ndim = {x.ndim}).')
        # 1-dimensional array (0-dimensional should not be here, but it could be handled)
        else:
            if w is None:
                invalid = np.isnan(x) | np.isinf(x)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = np.sum(~invalid)
                mean = np.mean(x[~invalid])
                var = np.mean((x[~invalid] - mean) ** 2)
                return mean, var, weight
            else:
                if w.ndim > 1 or len(w) != len(x):
                    raise ValueError('w has to have the same shape and size as x')
                invalid = np.isnan(x) | np.isinf(x) | np.isnan(w) | np.isinf(w)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = np.sum(w[~invalid])
                mean = np.average(x[~invalid], weights=w[~invalid])
                var = np.average((x[~invalid] - mean) ** 2, weights=w[~invalid])
                return mean, var, weight

    def update(self,
               x: Union[Number, np.ndarray, pd.Series],
               w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
               raise_if_nans: bool = False) -> OMeanVar:
        x = self._unwrap_if_possible(x)
        w = self._unwrap_if_possible(w)
        if isinstance(x, np.ndarray):
            other = OMeanVar(*self._mean_var_weight_of_np(x, w, raise_if_nans=raise_if_nans))
        else:
            other = OMeanVar(x, 0, w)
        self += other
        return self

    def __add__(self, other: OMeanVar) -> OMeanVar:
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

    def __iadd__(self, other: OMeanVar) -> OMeanVar:
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            self.mean = other.mean
            self.var = other.var
            self.weight = other.weight
            return self
        elif other.weight == 0:
            return self
        else:
            delta_mean = other.mean - self.mean
            delta_var = other.var - self.var
            ratio = other.weight / (self.weight + other.weight)
            self.mean = self.mean + delta_mean * ratio
            self.var = self.var + delta_var * ratio + delta_mean ** 2 * ratio * (1 - ratio)
            self.weight = self.weight + other.weight
            return self

    @property
    def std_dev(self) -> float:
        return np.sqrt(self.var)

    @property
    def unbiased_var(self) -> float:
        return self.var * (self.weight / (self.weight - 1))

    @property
    def unbiased_std_dev(self) -> float:
        return np.sqrt(self.unbiased_var)

