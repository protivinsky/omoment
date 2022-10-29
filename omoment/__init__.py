#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMoment package calculates moments of statistical distributions (means and variance) in online or distributed settings.

- Suitable for large data – works well with numpy and Pandas and in distributed setting.
- Moments calculated from different parts of data can be easily combined or updated for new data (supports addition
  of results).
- Objects are lightweight, calculation is done in numpy if possible.
- Weights for data can be provided.
- Invalid values (NaNs, infinities are omitted by default).

Typical application is calculation of means and variances of many chunks of data (corresponding to different groups
or to different parts of the distributed data), the results can be analyzed on level of the groups or easily
combined to get exact moments for the full dataset.

Basic example
-------------

.. code:: python

    from omoment import OMeanVar
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(12354)
    g = rng.integers(low=0, high=10, size=1000)
    x = g + rng.normal(loc=0, scale=10, size=1000)
    w = rng.exponential(scale=1, size=1000)

    # calculate overall moments
    OMeanVar(x, weight=w)
    # should give: OMeanVar(mean=4.6, var=108, weight=1.08e+03)

    # or calculate moments for every group
    df = pd.DataFrame({'g': g, 'x': x, 'w': w})
    omvs = df.groupby('g').apply(OMeanVar.of_frame, x='x', w='w')

    # and combine group moments to obtain the same overall results
    OMeanVar.combine(omvs)

    # addition is also supported
    omvs.loc[0] + omvs.loc[1]

At the moment, only univariate distributions are supported. Bivariate or even multivariate distributions can be
efficiently processed in a similar fashion, so the support for them might be added in the future. Moments of
multivariate distributions would also allow for linear regression estimation and other statistical methods
(such as PCA or regularized regression) to be calculated in a single pass through large distributed datasets.
"""

from .obase import OBase
from .omean import OMean
from .omeanvar import OMeanVar
from os import path

__all__ = ['OBase', 'OMean', 'OMeanVar']
__author__ = 'Tomas Protivinsky'
__version__ = "0.1.0"
