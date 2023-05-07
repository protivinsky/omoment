from __future__ import annotations
from numbers import Number
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List
from omoment import OBase, HandlingInvalid


class OReg(OBase):
    r"""Online estimator of univariate regression of two variables.

    Represents means, variances, covariance and total weight of a part of data. Two `OReg` objects can be added
    together to produce correct estimates for the combined dataset. Moments and weight are stored using `__slots__`
    to allow for lightweight objects that can be used in fairly large quantities even in pandas DataFrame (however
    they are still Pythonm objects, not numpy types).

    By default, invalid values are omitted in calculation; variances and covariance are based on `ddof = 0`,
    in agreement with numpy `std` method.
    """

    __slots__ = ['mean_x', 'mean_y', 'var_x', 'var_y', 'cov', 'weight']

    def __init__(self,
                 mean_x: Number = np.nan,
                 mean_y: Number = np.nan,
                 var_x: Number = np.nan,
                 var_y: Number = np.nan,
                 cov: Number = np.nan,
                 weight: Number = 0,
                 handling_invalid: HandlingInvalid = HandlingInvalid.Drop):

        # isn't this too slow?
        for s in self.__slots__:
            if not (isinstance(eval(s), Number)):
                raise TypeError(f'{s} has to be a number, provided type({s}) = {type(eval(s)).__name__}.')

        if weight and not (handling_invalid == HandlingInvalid.Keep):
            if weight < 0 or not np.isfinite(weight) or (weight > 0 and not (np.isfinite(mean_x)
                    and np.isfinite(mean_y) and np.isfinite(var_x) and np.isfinite(var_y) and np.isfinite(cov)
                    and var_x > -1e-10 and var_y > -1e-10 and abs(cov) < np.sqrt(var_x * var_y) + 1e-10)):
                if handling_invalid == HandlingInvalid.Raise:
                    if weight < 0 or not np.isfinite(weight):
                        raise ValueError(f'Invalid weight in OReg: weight = {weight}')
                    else:
                        raise ValueError(f'Invalid input values in OReg.')
                else:
                    mean_x = np.nan
                    mean_y = np.nan
                    var_x = np.nan
                    var_y = np.nan
                    cov = np.nan
                    weight = 0
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.var_x = var_x
        self.var_y = var_y
        self.cov = cov
        self.weight = weight

    @staticmethod
    def _calculate_of_np(x: np.ndarray,
                         y: np.ndarray,
                         w: Optional[np.ndarray] = None,
                         handling_invalid: HandlingInvalid = HandlingInvalid.Drop
                         ) -> Tuple[float, float, float, float, float, float]:
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError(f'x and y has to be a np.ndarray, their types are {type(x)} and {type(y)} '
                            f'respectively.')
        elif x.ndim > 1 or x.shape != y.shape:
            raise ValueError(f'Provided x, y arrays has to be 1-dimensional and of the same shape.')
        else:
            if w is None:
                w = np.ones_like(x)
            if handling_invalid != HandlingInvalid.Keep:
                invalid = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(w) | (w < 0.)
                if handling_invalid == HandlingInvalid.Raise and np.sum(invalid):
                    raise ValueError('x, w or y contains invalid values (nan or infinity).')
                w = w[~invalid]
                x = x[~invalid]
                y = y[~invalid]
            weight = np.sum(w)
            mean_x = np.average(x, weights=w)
            mean_y = np.average(y, weights=w)
            var_x = np.average((x - mean_x) ** 2, weights=w)
            var_y = np.average((y - mean_y) ** 2, weights=w)
            cov = np.average((x - mean_x) * (y - mean_y), weights=w)
            return mean_x, mean_y, var_x, var_y, cov, weight

    def update(self,
               x: Union[Number, np.ndarray, pd.Series],
               y: Union[Number, np.ndarray, pd.Series],
               w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
               handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OReg:
        x = self._unwrap_if_possible(x)
        y = self._unwrap_if_possible(y)
        w = self._unwrap_if_possible(w)
        if isinstance(x, np.ndarray):
            other = OReg(*self._calculate_of_np(x, y, w, handling_invalid=handling_invalid))
        else:
            other = OReg(x, y, 0., 0., 0., w, handling_invalid=handling_invalid)
        self += other
        return self

    def __iadd__(self, other: OReg) -> OReg:
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            self.mean_x = other.mean_x
            self.mean_y = other.mean_y
            self.var_x = other.var_x
            self.var_y = other.var_y
            self.cov = other.cov
            self.weight = other.weight
            return self
        elif other.weight == 0:
            return self
        else:
            delta_mean_x = other.mean_x - self.mean_x
            delta_mean_y = other.mean_y - self.mean_y
            delta_var_x = other.var_x - self.var_x
            delta_var_y = other.var_y - self.var_y
            delta_cov = other.cov - self.cov
            ratio = other.weight / (self.weight + other.weight)
            self.mean_x = self.mean_x + delta_mean_x * ratio
            self.mean_y = self.mean_y + delta_mean_y * ratio
            self.var_x = self.var_x + delta_var_x * ratio + delta_mean_x ** 2 * ratio * (1 - ratio)
            self.var_y = self.var_y + delta_var_y * ratio + delta_mean_y ** 2 * ratio * (1 - ratio)
            self.cov = self.cov + delta_cov * ratio + delta_mean_x * delta_mean_y * ratio * (1 - ratio)
            self.weight = self.weight + other.weight
            return self

    @property
    def std_dev_x(self) -> float:
        return np.sqrt(self.var_x)

    @property
    def std_dev_y(self) -> float:
        return np.sqrt(self.var_y)

    @property
    def corr(self) -> float:
        return self.cov / (self.std_dev_x * self.std_dev_y)

    @property
    def alpha(self) -> float:
        return self.mean_y - self.beta * self.mean_x

    @property
    def beta(self) -> float:
        return self.cov / self.var_x

    @classmethod
    def of_frame(cls, data: pd.DataFrame, x: str, y: str, w: Optional[str] = None,
                 handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OReg:
        """Convenience function for calculation OReg of pandas DataFrame.

        Args:
            data: input DataFrame
            x: name of column with x variable
            y: name of column with y variable
            w: name of column with weights (optional)
            handling_invalid: How to handle invalid values in calculation ['drop', 'keep', 'raise'], default value
             'drop'. Provided either as enum or its string representation.

        Returns:
            OReg object

        """
        if w is None:
            return cls.compute(data[x].values, data[y].values, handling_invalid=handling_invalid)
        else:
            return cls.compute(data[x].values, data[y].values, data[w].values, handling_invalid=handling_invalid)

    @staticmethod
    def of_groupby(data: pd.DataFrame,
                   g: Union[str, List[str]],
                   x: str, y: str, w: Optional[str] = None,
                   handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> pd.Series[OReg]:
        orig_len = len(data)
        cols = (g if isinstance(g, list) else [g]) + ([x, y] if w is None else [x, y, w])
        data = data[cols]
        if handling_invalid == HandlingInvalid.Keep:
            data = data.copy()
        else:
            data = data[np.isfinite(data).all(1)].copy()
            if handling_invalid == HandlingInvalid.Raise and len(data) < orig_len:
                raise ValueError('data contains invalid values (nan or infinity).')
        if w is None:
            w = '_w'
            data[w] = 1
            data = data.rename(columns={x: '_xw', y: '_yw'})
            data['_xxw'] = data['_xw'] ** 2
            data['_yyw'] = data['_yw'] ** 2
            data['_xyw'] = data['_xw'] * data['_yw']
        else:
            data['_xw'] = data[x] * data[w]
            data['_yw'] = data[y] * data[w]
            data['_xxw'] = data[x] ** 2 * data[w]
            data['_yyw'] = data[y] ** 2 * data[w]
            data['_xyw'] = data[x] * data[y] * data[w]
        agg = data.groupby(g)[['_xw', '_yw', '_xxw', '_yyw', '_xyw', w]].sum()
        agg['mean_x'] = agg['_xw'] / agg[w]
        agg['mean_y'] = agg['_yw'] / agg[w]
        agg['var_x'] = (agg['_xxw'] - agg['mean_x'] ** 2 * agg[w]) / agg[w]
        agg['var_y'] = (agg['_yyw'] - agg['mean_y'] ** 2 * agg[w]) / agg[w]
        agg['cov'] = (agg['_xyw'] - agg['mean_x'] * agg['mean_y'] * agg[w]) / agg[w]
        res = agg.apply(lambda row: OReg(row['mean_x'], row['mean_y'], row['var_x'], row['var_y'], row['cov'], row[w]),
                        axis=1)
        return res

    @staticmethod
    def get_mean_x(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.mean_x

    @staticmethod
    def get_mean_y(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.mean_y

    @staticmethod
    def get_weight(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.weight

    @staticmethod
    def get_var_x(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.var_x

    @staticmethod
    def get_std_dev_x(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.std_dev_x

    @staticmethod
    def get_var_y(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.var_y

    @staticmethod
    def get_std_dev_y(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.std_dev_y

    @staticmethod
    def get_cov(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.cov

    @staticmethod
    def get_corr(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.corr

    @staticmethod
    def get_alpha(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.alpha

    @staticmethod
    def get_beta(oreg: OReg):
        """Convenience function to be used as a lambda."""
        return oreg.beta

    # Explicit override to allow for intellisense
    @classmethod
    def compute(cls,
                x: Union[Number, np.ndarray, pd.Series],
                y: Union[Number, np.ndarray, pd.Series],
                w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
                handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OReg:
        oreg = cls()
        oreg.update(x=x, y=y, w=w, handling_invalid=handling_invalid)
        return oreg

