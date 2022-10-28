from __future__ import annotations
import math
from numbers import Number
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple, Iterable
from abc import ABC, abstractmethod


class OBase(ABC):
    """Base class for moment calculating online estimators."""
    @abstractmethod
    def update(self,
               x: Union[Number, np.ndarray, pd.Series],
               w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
               raise_if_nans: bool = False) -> OBase:
        ...

    @abstractmethod
    def _validate(self) -> None:
        ...

    @abstractmethod
    def __add__(self, other: OBase) -> OBase:
        ...

    @abstractmethod
    def __iadd__(self, other: OBase) -> OBase:
        ...

    def __eq__(self, other: OBase) -> bool:
        return all([self.__getattribute__(s) == other.__getattribute__(s) for s in self.__slots__])

    def is_close(self, other: OBase, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        return all([math.isclose(self.__getattribute__(s), other.__getattribute__(s), rel_tol=rel_tol, abs_tol=abs_tol)
                    for s in self.__slots__])

    def __repr__(self) -> str:
        fields = ', '.join([f'{s}={self.__getattribute__(s)}' for s in self.__slots__])
        return f'{self.__class__.__name__}({fields})'

    def __str__(self) -> str:
        fields = ', '.join([f'{s}={self.__getattribute__(s):.3g}' for s in self.__slots__])
        return f'{self.__class__.__name__}({fields})'

    def to_dict(self) -> Dict[str, Number]:
        return {s: self.__getattribute__(s) for s in self.__slots__}

    def to_tuple(self) -> Tuple[Number]:
        return tuple(self.__getattribute__(s) for s in self.__slots__)

    @classmethod
    def of_dict(cls, d: Dict[str, Number]) -> OBase:
        return cls(**d)

    @classmethod
    def of_tuple(cls, t: Tuple[Number]) -> OBase:
        return cls(*t)

    def copy(self) -> OBase:
        return self.__class__(*(self.__getattribute__(s) for s in self.__slots__))

    @classmethod
    def combine(cls, first: Union[OBase, Iterable[OBase]], second: Optional[OBase] = None) -> OBase:
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
    def _unwrap_if_possible(x: Union[Number, np.ndarray, pd.Series]) -> Union[Number, np.ndarray]:
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(x, np.ndarray) and x.size == 1:
            x = float(x)
        return x


