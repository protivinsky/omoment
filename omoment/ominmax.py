#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Union, Optional, Tuple, List
from numbers import Number
import numpy as np
import pandas as pd
from omoment import OBase, HandlingInvalid


class OMinMax(OBase):
    """
    Only estimator for minimum, maximum and number of points.

    Not really moments, but it is useful and shares the same interface. Does not use weights.
    """

    __slots__ = ['min', 'max', 'n']

    def __init__(self,
                 min: Number = np.inf,
                 max: Number = -np.inf,
                 n: int = 0,
                 handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMinMax:
        """Represents the minimum, maximum and number of points of chunk of data.

        Raises:
            TypeError: If min, max or n is not a number.
        """
        # The shadowing of built-in names is intentional: I do not like it either, but some of OBase methods require
        # that the keyword arguments have the same names as the attributes. Renaming the attributes would only lead
        # to more confusion and typos; the built-ins are not obviously used in the __init__ method.
        if not(isinstance(min, Number) and isinstance(max, Number) and isinstance(n, Number)):
            raise TypeError(f'Min, max and n has to be numbers, got {type(min)}, {type(max)} and {type(n)}.')
        if handling_invalid != HandlingInvalid.Keep:
            if not(np.isfinite(min) and np.isfinite(max) and np.isfinite(n) and max >= min and n >= -1e-10):
                if handling_invalid == HandlingInvalid.Raise:
                    raise ValueError(f'Invalid values in OMinMax.__init__: {min}, {max}, {n}')
                else:
                    min = np.inf
                    max = -np.inf
                    n = 0
        self.min = min
        self.max = max
        self.n = n

    def update(self, x: Union[Number, np.ndarray, pd.Series],
               handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMinMax:
        """
        Update the values based on new data, provided either as a single number or an array-like object.
        """
        x = self._unwrap_if_possible(x)
        if not isinstance(x, np.ndarray):
            x = np.ndarray([x])
        elif x.ndim != 1:
            raise ValueError(f'Array has to be 1-dimensional, got {x.ndim}-dimensional.')
        if handling_invalid != HandlingInvalid.Keep:
            invalid = ~np.isfinite(x)
            if handling_invalid == HandlingInvalid.Raise and np.any(invalid):
                raise ValueError(f'Invalid values in OMinMax.update: {x[invalid]}')
            x = x[~invalid]
        self.min = min(self.min, np.min(x))
        self.max = max(self.max, np.max(x))
        self.n += len(x)  # shall this be counted together with invalid values?
        return self

    def __iadd__(self, other: OMinMax) -> OMinMax:
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}.')
        if self.n == 0:
            self.min = other.min
            self.max = other.max
            self.n = other.n
            return self
        elif other.n == 0:
            return self
        else:
            self.min = min(self.min, other.min)
            self.max = max(self.max, other.max)
            self.n += other.n
            return self

    def __nonzero__(self):
        return self.n.__nonzero()

    @classmethod
    def of_frame(cls, data: pd.DataFrame, x: str,
                 handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMinMax:
        """
        Create an OMinMax based on a dataframe and a column name.
        """
        return cls.compute(data[x], handling_invalid=handling_invalid)

    @staticmethod
    def of_groupby(data: pd.DataFrame, g: Union[str, List[str]], x: str,
                   handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> pd.Series[OMinMax]:
        """Optimized version of OMinMax for large number of groups in data (so avoids costly `df.groupby -> apply`)."""
        orig_len = len(data)
        cols = (g if isinstance(g, list) else [g]) + [x]
        data = data[cols]
        if handling_invalid == HandlingInvalid.Keep:
            data = data.copy()
        else:
            data = data[np.isfinite(data).all(1)].copy()
            if handling_invalid == HandlingInvalid.Raise and len(data) < orig_len:
                raise ValueError('x contains invalid values (nan or infinity).')
        agg = data.groupby(g)[x].agg(['min', 'max', 'count'])
        res = agg.apply(lambda row: OMinMax(row['min'], row['max'], row['count']), axis=1)
        return res

    @staticmethod
    def get_min(omm: OMinMax):
        """Convenience method to get the minimum value."""
        return omm.min

    @staticmethod
    def get_max(omm: OMinMax):
        """Convenience method to get the maximum value."""
        return omm.max

    @staticmethod
    def get_n(omm: OMinMax):
        """Convenience method to get the number of points."""
        return omm.n

    # Explicit override to allow for intellisense
    @classmethod
    def compute(cls,
                x: Union[Number, np.ndarray, pd.Series],
                handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OMinMax:
        omm = cls()
        omm.update(x=x, handling_invalid=handling_invalid)
        return omm
