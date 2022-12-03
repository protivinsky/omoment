#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from numbers import Number
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple, Iterable
from abc import ABC, abstractmethod
from enum import Enum


class HandlingInvalid(str, Enum):
    Drop = 'drop'
    Keep = 'keep'
    Raise = 'raise'


class OBase(ABC):
    """
    Base class for moment calculating online estimators for univariate distributions.

    Provides basic functionality such as:

    - equality and near-equality (for floats, `is_close`)
    - string representation
    - conversion objects to dictionaries or tuples (convenient for serialization)
    - combination of objects via addition
    - copying of objects

    """
    @abstractmethod
    def update(self,
               x: Union[Number, np.ndarray, pd.Series],
               w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
               handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OBase:
        """
        Update the object based on new data. Subclasses have to implement how to aggregate the new data.
        """
        ...

    @classmethod
    def compute(cls, *args, **kwargs):
        """
        Shortcut for initialization of an empty object and its update based on data.
        """
        ob = cls()
        ob.update(*args, **kwargs)
        return ob

    @abstractmethod
    def __iadd__(self, other: OBase) -> OBase:
        """
        In-place addition, mutates the self object.
        """
        ...

    def __add__(self, other: OBase) -> OBase:
        """
        Addition of two objects. Produces a new object.
        """
        ob = self.__class__()
        ob += self
        ob += other
        return ob

    def __eq__(self, other: OBase) -> bool:
        """
        Test of equality by comparing all elements.
        """
        return all([self.__getattribute__(s) == other.__getattribute__(s) for s in self.__slots__])

    def is_close(self, other: OBase, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """
        Test of near equality: all values are compared with `math.isclose` and provided relative and absolute
        tolerance.
        """
        return all([math.isclose(self.__getattribute__(s), other.__getattribute__(s), rel_tol=rel_tol, abs_tol=abs_tol)
                    for s in self.__slots__])

    def __repr__(self) -> str:
        fields = ', '.join([f'{s}={self.__getattribute__(s)}' for s in self.__slots__])
        return f'{self.__class__.__name__}({fields})'

    def __str__(self) -> str:
        fields = ', '.join([f'{s}={self.__getattribute__(s):.3g}' for s in self.__slots__])
        return f'{self.__class__.__name__}({fields})'

    def to_dict(self) -> Dict[str, Number]:
        """
        Convert the object to dictionary, attribute names are used as keys (for JSON serialization etc.).
        """
        return {s: self.__getattribute__(s) for s in self.__slots__}

    def to_tuple(self) -> Tuple[Number]:
        """
        Convert the values of attributes to a tuple.
        """
        return tuple(self.__getattribute__(s) for s in self.__slots__)

    @classmethod
    def of_dict(cls, d: Dict[str, Number]) -> OBase:
        """
        Reconstruct the object from a dictionary (inverse of `to_dict` method).
        """
        return cls(**d)

    @classmethod
    def of_tuple(cls, t: Tuple[Number]) -> OBase:
        """
        Reconstruct the object from a tuple (inverse of `to_tuple` method).
        """
        return cls(*t)

    def copy(self) -> OBase:
        """
        Create a new object with identical values.
        """
        return self.__class__(*(self.__getattribute__(s) for s in self.__slots__))

    @classmethod
    def combine(cls, first: Union[OBase, Iterable[OBase]], second: Optional[OBase] = None) -> OBase:
        """
        Either combine two objects together or merge an iterable of objects. The logic for correct combination has
        to be implemented in `__add__` and `__iadd__` of subclasses.
        """
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
    def _unwrap_if_possible(x: Union[Number, np.ndarray, pd.Series]) -> Union[Number, np.ndarray]:
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(x, np.ndarray) and x.size == 1:
            x = float(x)
        return x


