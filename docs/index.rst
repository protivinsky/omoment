
Welcome to OMoment’s documentation!
***********************************

A zde je nejaky text.

**class omoment.OBase**

   Base class for moment calculating online estimators.

**class omoment.OMean(mean: Union[Number, ndarray, Series] = nan,
weight: Optional[Union[Number, ndarray, Series]] = None)**

   Online calculation of (weighted) mean.

   Specifically, a Box represents the Cartesian product of n closed
   intervals. Each interval has the form of one of [a, b], (-\infty,
   b], [a, \infty), or (-\infty, \infty).

**OMean.update(x: Union[Number, ndarray, Series], w:
Optional[Union[Number, ndarray, Series]] = None, raise_if_nans: bool =
False) -> `OMean <#omoment.OMean>`_**

   Update the moments by adding some values; NaNs are removed both
   from values and from weights.

   Args:
      x: Values to add to the estimator. w: Weights for new values. If
      provided, has to have the same length as x. raise_if_nans: If
      true, raises an error if there are NaNs in data. Otherwise, they
      are silently removed.

**class omoment.OMeanVar(mean: Union[Number, ndarray, Series] = nan,
var: Optional[Number] = None, weight: Optional[Union[Number, ndarray,
Series]] = None)**
