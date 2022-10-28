import math
import numpy as np
import pandas as pd
import typing
from abc import ABC, abstractmethod


class OBase(ABC):
    """Base class for moment calculating online estimators."""
    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def _validate(self):
        ...

    @abstractmethod
    def __add__(self, other):
        ...

    @abstractmethod
    def __iadd__(self, other):
        ...

    def __eq__(self, other):
        return all([self.__getattribute__(s) == other.__getattribute__(s) for s in self.__slots__])

    def is_close(self, other, rel_tol=1e-09, abs_tol=0.0):
        return all([math.isclose(self.__getattribute__(s), other.__getattribute__(s), rel_tol=rel_tol, abs_tol=abs_tol)
                    for s in self.__slots__])

    def __repr__(self):
        fields = ', '.join([f'{s}={self.__getattribute__(s)}' for s in self.__slots__])
        return f'{self.__class__.__name__}({fields})'

    def __str__(self):
        fields = ', '.join([f'{s}={self.__getattribute__(s):.3g}' for s in self.__slots__])
        return f'{self.__class__.__name__}({fields})'

    def to_dict(self):
        return {s: self.__getattribute__(s) for s in self.__slots__}

    def to_tuple(self):
        return tuple(self.__getattribute__(s) for s in self.__slots__)

    @classmethod
    def of_dict(cls, d):
        return cls(**d)

    @classmethod
    def of_tuple(cls, t):
        return cls(*t)

    def copy(self):
        return self.__class__(*(self.__getattribute__(s) for s in self.__slots__))

    @classmethod
    def combine(cls, first, second=None):
        """Combine either an iterable of OClasses or two OClass objects together."""
        if second is None:
            if isinstance(first, pd.Series):
                first = first.values
            result = cls()
            for other in first:
                result += other
            return result
        else:
            if not (isinstance(first, cls) and isinstance(second, cls)):
                raise TypeError(f'Both first and second arguments have to be instances of {cls}.')
            return first + second

    @staticmethod
    def _unwrap_if_possible(x):
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(x, np.ndarray) and x.size == 1:
            x = float(x)
        return x


