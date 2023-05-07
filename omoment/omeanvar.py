#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from numbers import Number
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List
from omoment import OMean, HandlingInvalid


class OMeanVar(OMean):
    r"""Online estimator of weighted mean and variance.

    Represents mean, variance and weight of a part of data. Two `OMeanVar` objects can be added together to produce
    correct estimates for overall dataset. Mean, variance and weight are stored using `__slots__` to allow for
    lightweight objects that can be used in large quantities even in pandas DataFrame (however they are still Python
    objects, not numpy types).

    Most methods are fairly permissive, allowing to work on numbers, numpy arrays or pandas DataFrames. By default,
    invalid values are omitted from data (NaNs, infinities and negative weights). Variance in `OMeanVar` is based on
    `ddof = 0`, in agreement with numpy `std` method.

    Addition of :math:`\mathrm{OMeanVar}(m_1, v_1, w_1)` and :math:`\mathrm{OMeanVar}(m_2, v_2, w_2)` is calculated as:

    .. math::
        :nowrap:

        \begin{gather*}
        \delta_m = m_2 - m_1\\
        \delta_v = v_2 - v_1\\
        w_N = w_1 + w_2\\
        r = \frac{w_2}{w_N}\\
        m_N = m_1 + \delta_m \frac{w_2}{w_N}\\
        v_N = v_1 + \delta_v r + \delta_m^2 r (1 - r)
        \end{gather*}

    Where subscript N denotes the new values produced by the addition.
    """

    __slots__ = ['mean', 'var', 'weight']

    def __init__(self,
                 mean: Number = np.nan,
                 var: Number = np.nan,
                 weight: Number = 0,
                 handling_invalid: HandlingInvalid = HandlingInvalid.Drop):
        """Creates a representation of mean, variance and weight of a part of data. To estimate OMeanVar directly
        from data, use OMeanVar.compute class method.

        Raises:
            ValueError: if handling invalid is HandlingInvalid.Raise (or 'raise') and invalid values are provided
             in the constructor (such as negative weight or positive weight and infinite mean).

        """
        if not (isinstance(mean, Number) and isinstance(var, Number) and isinstance(weight, Number)):
            raise TypeError(f'Mean, var and weight have to be numbers in OMeanVar, provided type(mean) = '
                            f'{type(mean).__name__}, type(var) = {type(var).__name__}, type(weight) = '
                            f'{type(weight).__name__}.')
        if weight and not (handling_invalid == HandlingInvalid.Keep):
            if weight < 0 or not np.isfinite(weight):
                if handling_invalid == HandlingInvalid.Raise:
                    raise ValueError(f'Invalid weight in OMeanVar: weight = {weight}')
                else:
                    mean = np.nan
                    var = np.nan
                    weight = 0
            elif weight > 0 and not (np.isfinite(mean) and np.isfinite(var) and (var > -1e-10)):
                if handling_invalid == HandlingInvalid.Raise:
                    raise ValueError(f'Invalid mean or var with positive weight in OMeanVar: mean = {mean}, '
                                     f'var = {var}')
                else:
                    mean = np.nan
                    var = np.nan
                    weight = 0
        self.mean = mean
        self.var = var
        self.weight = weight

    @staticmethod
    def _mean_var_weight_of_np(x: np.ndarray,
                               w: Optional[np.ndarray] = None,
                               handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> Tuple[float, float, float]:
        """
        Helper function to calculate mean, variance and weight from numpy arrays.
        """
        # check if we have np.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError(f'x has to be a np.ndarray, it is {type(x)}.')
        # more than 1-dimensional array, throw an exception
        elif x.ndim > 1:
            raise ValueError(f'Provided np.ndarray has to be 1-dimensional (x.ndim = {x.ndim}).')
        # 1-dimensional array (0-dimensional should not be here, but it could be handled)
        else:
            if w is None:
                if handling_invalid == HandlingInvalid.Keep:
                    invalid = np.full(len(x), fill_value=False)
                else:
                    invalid = ~np.isfinite(x)
                    if handling_invalid == HandlingInvalid.Raise and np.sum(invalid):
                        raise ValueError('x contains invalid values (nan or infinity).')
                weight = np.sum(~invalid)
                mean = np.mean(x[~invalid])
                var = np.mean((x[~invalid] - mean) ** 2)
                return mean, var, weight
            else:
                if w.ndim > 1 or len(w) != len(x):
                    raise ValueError('w has to have the same shape and size as x')
                if handling_invalid == HandlingInvalid.Keep:
                    invalid = np.full(len(x), fill_value=False)
                else:
                    invalid = ~np.isfinite(x) | ~np.isfinite(w) | (w < 0)
                    if handling_invalid == HandlingInvalid.Raise and np.sum(invalid):
                        raise ValueError('x contains invalid values (nan or infinity).')
                weight = np.sum(w[~invalid])
                mean = np.average(x[~invalid], weights=w[~invalid])
                var = np.average((x[~invalid] - mean) ** 2, weights=w[~invalid])
                return mean, var, weight

    def update(self,
               x: Union[Number, np.ndarray, pd.Series],
               w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
               handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMeanVar:
        """Update the moments by adding new values.

        Can be either single values or batch of data in numpy arrays. In the latter case, moments are first estimated
        on the new data and the moments for old and new data are combined. Invalid values and negative weights are
        omitted by default. The calculated variance assumes zero degrees of freedom, `OMeanVar` has properties
        `unbiased_var` and `unbiased_std_dev` based on dof = 1.

        Args:
            x: Values to add to the current estimate.
            w: Weights for the values. If provided, has to have the same length as x.
            handling_invalid: How to handle invalid values in calculation ['drop', 'keep', 'raise'], default value
             'drop'. Provided either as enum or its string representation.

        Returns:
            The same OMeanVar object updated for the new data.

        Raises:
            ValueError: `if raise_if_nans` is True and there are invalid values (NaNs, infinities or negative weights)
             in data.
            TypeError: if values x or w have more than one dimension or if they are of different size.

        """
        x = self._unwrap_if_possible(x)
        w = self._unwrap_if_possible(w)
        if isinstance(x, np.ndarray):
            other = OMeanVar(*self._mean_var_weight_of_np(x, w, handling_invalid=handling_invalid))
        else:
            other = OMeanVar(x, 0, w, handling_invalid=handling_invalid)
        self += other
        return self

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
        r"""
        Estimate of standard deviation, calculated as :math:`\sqrt{\mathrm{Var}}`. Based on ddof = 0, the same default
        as in numpy `std` method.
        """
        return np.sqrt(self.var)

    @property
    def unbiased_var(self) -> float:
        """Estimate of unbiased variance based on ddof = 1 (suitable for unweighted data).
        """
        return self.var * (self.weight / (self.weight - 1))

    @property
    def unbiased_std_dev(self) -> float:
        """Estimate of unbiased standard deviation based on ddof = 1 (suitable for unweighted data).
        """
        return np.sqrt(self.unbiased_var)

    @staticmethod
    def of_groupby(data: pd.DataFrame,
                   g: Union[str, List[str]],
                   x: str, w: Optional[str] = None,
                   handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> pd.Series[OMean]:
        """Optimized version for calculation of means of **large number of groups** in data.

        Avoids slower groupby -> apply workflow and uses optimized aggregation functions only. The function is about
        5x faster on testing dataset with 10,000,000 rows and 100,000 groups.

        Args:
            data: input DataFrame
            g: name of column containing group keys; can be also a list of multiple column names
            x: name of column with values to calculated mean of
            w: name of column with weights (optional)
            handling_invalid: How to handle invalid values in calculation ['drop', 'keep', 'raise'], default value
             'drop'. Provided either as enum or its string representation.

        Returns:
            pandas Series indexed by group values g and containing estimated OMeanVar objects
        """
        orig_len = len(data)
        cols = (g if isinstance(g, list) else [g]) + ([x] if w is None else [x, w])
        data = data[cols]
        if handling_invalid == HandlingInvalid.Keep:
            data = data.copy()
        else:
            data = data[np.isfinite(data).all(1)].copy()
            if handling_invalid == HandlingInvalid.Raise and len(data) < orig_len:
                raise ValueError('x or w contains invalid values (nan or infinity).')
        if w is None:
            w = '_w'
            data[w] = 1
            data = data.rename(columns={x: '_xw'})
            data['_xxw'] = data['_xw'] ** 2
        else:
            data['_xw'] = data[x] * data[w]
            data['_xxw'] = data[x] ** 2 * data[w]
        agg = data.groupby(g)[['_xw', '_xxw', w]].sum()
        agg['mean'] = agg['_xw'] / agg[w]
        agg['var'] = (agg['_xxw'] - agg['mean'] ** 2 * agg[w]) / agg[w]
        res = agg.apply(lambda row: OMeanVar(row['mean'], row['var'], row[w]), axis=1)
        return res

    @staticmethod
    def get_var(om: OMeanVar):
        """Convenience function to be used as a lambda."""
        return om.var

    @staticmethod
    def get_std_dev(om: OMeanVar):
        """Convenience function to be used as a lambda."""
        return om.std_dev

    @staticmethod
    def get_unbiased_var(om: OMeanVar):
        """Convenience function to be used as a lambda."""
        return om.unbiased_var

    @staticmethod
    def get_unbiased_std_dev(om: OMeanVar):
        """Convenience function to be used as a lambda."""
        return om.unbiased_std_dev

    # Explicit override to allow for intellisense
    @classmethod
    def compute(cls,
                x: Union[Number, np.ndarray, pd.Series],
                w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
                handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMeanVar:
        omv = cls()
        omv.update(x=x, w=w, handling_invalid=handling_invalid)
        return omv

