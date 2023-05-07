#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Union, Optional, Tuple, List
from numbers import Number
import numpy as np
import pandas as pd
from omoment import OBase, HandlingInvalid


class OMean(OBase):
    r"""Online estimator of weighted mean.

    Represents mean and weight of a part of data. Two `OMean` objects can be added together to produce correct
    estimates for larger dataset. Mean and weight are stored using `__slots__` to allow for lightweight objects that
    can be used in large quantities even in pandas DataFrames.

    Most methods are fairly permissive, allowing to work on numbers, numpy arrays or pandas DataFrames. By default,
    invalid values are omitted from data (NaNs, infinities and negative weights).

    Addition of :math:`\mathrm{OMean}(m_1, w_1)` and :math:`\mathrm{OMean}(m_2, w_2)` is calculated as:

    .. math::
        :nowrap:

        \begin{gather*}
        \delta = m_2 - m_1\\
        w_N = w_1 + w_2\\
        m_N = m_1 + \delta \frac{w_2}{w_N}
        \end{gather*}

    Where subscript N denotes the new values produced by the addition.
    """

    __slots__ = ['mean', 'weight']

    def __init__(self,
                 mean: Number = np.nan,
                 weight: Number = 0,
                 handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMean:
        """Creates a representation of mean and weight of a part of data. To estimate OMean directly
        from data, use OMean.compute class method.

        Raises:
            ValueError: if handling invalid is HandlingInvalid.Raise (or 'raise') and invalid values are provided
             in the constructor (such as negative weight or positive weight and infinite mean).

        """
        if not (isinstance(mean, Number) and isinstance(weight, Number)):
            raise TypeError(f'Mean and weight have to be numbers in OMean, provided type(mean) ='
                            f' {type(mean).__name__}, type(weight) = {type(weight).__name__}.')
        if weight and not (handling_invalid == HandlingInvalid.Keep):
            if weight < 0 or not np.isfinite(weight):
                if handling_invalid == HandlingInvalid.Raise:
                    raise ValueError(f'Invalid weight in OMean: weight = {weight}')
                else:
                    mean = np.nan
                    weight = 0
            elif weight > 0 and not np.isfinite(mean):
                if handling_invalid == HandlingInvalid.Raise:
                    raise ValueError(f'Invalid mean with positive weight in OMean: mean = {mean}')
                else:
                    mean = np.nan
                    weight = 0
        self.mean = mean
        self.weight = weight

    @staticmethod
    def _mean_weight_of_np(x: np.ndarray,
                           w: Optional[np.ndarray] = None,
                           handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> Tuple[float, float]:
        """
        Helper function to calculate mean and weight from numpy arrays.
        """
        # check if we have np.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError(f'x has to be a np.ndarray, it is {type(x)}.')
        # more than 1-dimensional array, throw an exception
        elif x.ndim > 1:
            raise TypeError(f'Provided np.ndarray has to be 1-dimensional (x.ndim = {x.ndim}).')
        # 1-dimensional array (0-dimensional should not be there, but it should not matter anyway
        else:
            if w is None:
                if handling_invalid == HandlingInvalid.Keep:
                    return np.mean(x), len(x)
                invalid = ~np.isfinite(x)
                if handling_invalid == HandlingInvalid.Raise and np.sum(invalid):
                    raise ValueError('x contains invalid values (nan or infinity).')
                return np.mean(x[~invalid]), np.sum(~invalid)
            else:
                if w.ndim > 1 or len(w) != len(x):
                    raise ValueError('w has to have the same shape and size as x')
                if handling_invalid == HandlingInvalid.Keep:
                    return np.average(x, weights=w), np.sum(w)
                invalid = np.isnan(x) | np.isinf(x) | np.isnan(w) | np.isinf(w) | (w < 0)
                if handling_invalid == HandlingInvalid.Raise and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = np.sum(w[~invalid])
                mean = np.average(x[~invalid], weights=w[~invalid])
                return mean, weight

    def update(self,
               x: Union[Number, np.ndarray, pd.Series],
               w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
               handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMean:
        """Update the moments by adding new values.

        Can be either single values or batch of data in numpy arrays. In the latter case, moments are first estimated
        on the new data and the moments for old and new data are combined. Invalid values and negative weights are
        omitted by default.

        Args:
            x: Values to add to the current estimate.
            w: Weights for the values. If provided, has to have the same length as x.
            handling_invalid: How to handle invalid values in calculation ['drop', 'keep', 'raise'], default value
             'drop'. Provided either as enum or its string representation.

        Returns:
            The same OMean object updated for the new data.

        Raises:
            ValueError: `if raise_if_nans` is True and there are invalid values (NaNs, infinities or negative weights)
             in data.
            TypeError: if values x or w have more than one dimension or if they are of different size.

        """
        x = self._unwrap_if_possible(x)
        w = self._unwrap_if_possible(w)
        if isinstance(x, np.ndarray):
            x, w = self._mean_weight_of_np(x, w, handling_invalid=handling_invalid)
        self += OMean(x, w, handling_invalid=handling_invalid)
        return self

    def __iadd__(self, other: OMean) -> OMean:
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

    def __nonzero__(self) -> bool:
        """
        OMean is zero if it has zero weight: it behaves as zero element under the addition.
        """
        return self.weight.__nonzero__()

    @classmethod
    def of_frame(cls, data: pd.DataFrame, x: str, w: Optional[str] = None,
                 handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMean:
        """Convenience function for calculation OMean of pandas DataFrame.

        Args:
            data: input DataFrame
            x: name of column with values to calculated mean of
            w: name of column with weights (optional)
            handling_invalid: How to handle invalid values in calculation ['drop', 'keep', 'raise'], default value
             'drop'. Provided either as enum or its string representation.

        Returns:
            OMean object

        """
        if w is None:
            return cls.compute(data[x].values, handling_invalid=handling_invalid)
        else:
            return cls.compute(data[x].values, data[w].values, handling_invalid=handling_invalid)

    @staticmethod
    def of_groupby(data: pd.DataFrame,
                   g: Union[str, List[str]],
                   x: str, w: Optional[str] = None,
                   handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> pd.Series[OMean]:
        """Optimized version for calculation of means of **large number of groups** in data.

        Avoids slower "groupby -> apply" workflow and uses optimized aggregation functions only. The function is about
        4x faster on testing dataset with 10,000,000 rows and 100,000 groups.

        Args:
            data: input DataFrame
            g: name of column containing group keys; can be also a list of multiple column names
            x: name of column with values to calculated mean of
            w: name of column with weights (optional)
            handling_invalid: How to handle invalid values in calculation ['drop', 'keep', 'raise'], default value
             'drop'. Provided either as enum or its string representation.

        Returns:
            pandas Series indexed by group values g and containing estimated OMean objects

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
        else:
            data['_xw'] = data[x] * data[w]
        agg = data.groupby(g)[['_xw', w]].sum()
        agg['mean'] = agg['_xw'] / agg[w]
        res = agg.apply(lambda row: OMean(row['mean'], row[w]), axis=1)
        return res

    @staticmethod
    def get_mean(om: OMean):
        """Convenience function to be used as a lambda."""
        return om.mean

    @staticmethod
    def get_weight(om: OMean):
        """Convenience function to be used as a lambda."""
        return om.weight

    # Explicit override to allow for intellisense
    @classmethod
    def compute(cls,
                x: Union[Number, np.ndarray, pd.Series],
                w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
                handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMean:
        om = cls()
        om.update(x=x, w=w, handling_invalid=handling_invalid)
        return om
