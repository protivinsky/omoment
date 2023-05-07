#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OMoment package calculates moments of statistical distributions (means, variances, covariance) in online or
distributed settings for univariate and bivariate distributions.

- Suitable for large data – works well with numpy and Pandas and in distributed setting.
- Moments calculated from different parts of data can be easily combined or updated for new data (supports addition
  of results).
- Objects are lightweight, calculation is done in numpy if possible.
- Weights for data can be provided.
- Invalid values (NaNs, infinities are omitted by default).

Typical application is calculation of means and variances (or even correlation of two variables) of many chunks of data
(corresponding to different groups or to different parts of the distributed data), the results can be analyzed on level
of the groups or easily combined to get exact moments for the full dataset.

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
    OMeanVar.compute(x, w)
    # should give: OMeanVar(mean=4.6, var=108, weight=1.08e+03)

    # or calculate moments for every group
    df = pd.DataFrame({'g': g, 'x': x, 'w': w})
    omvs = df.groupby('g').apply(OMeanVar.of_frame, x='x', w='w')

    # and combine group moments to obtain the same overall results
    OMeanVar.combine(omvs)

    # addition is also supported
    omvs.loc[0] + omvs.loc[1]

At the moment, univariate and bivariate distributions are supported. Bivariate distributions allow for fast linear
regression with two variables (and constant) calculation. Even multivariate distributions can be
efficiently processed in a similar fashion, so the support for them might be added in the future. Moments of
multivariate distributions would also allow for linear regression estimation and other statistical methods
(such as PCA or regularized regression) to be calculated in a single pass through large distributed datasets.

Similar packages
----------------

OMoment package aims for fast calculation of weighted distribution moments (mean and variance at the moment),
great compatibility with numpy and pandas and suitability for distributed datasets (composability of results).
I have not found a package that would satisfy this, even though similar packages indeed exist.

RunStats
........

`RunStats
<https://grantjenks.com/docs/runstats/>`_ package calculates several moments of univariate distribution (including skewness and kurtosis)
and a few other statistics (min and max) and the results can be combined together. In addition, it provides Regression
object for bivariate statistics. It does not support weights and the calculation was more than 100x slower in my
testing (admittedly I am not sure if I used cython support correctly).

.. code:: python

    import numpy as np
    from omoment import OMeanVar
    from runstats import Statistics
    import time

    rng = np.random.Generator(np.random.PCG64(12345))
    x = rng.normal(size=1_000_000)

    start = time.time()
    omv = OMeanVar.compute(x)
    end = time.time()
    print(f'{end - start:.3g} seconds')
    # 0.0146 seconds

    start = time.time()
    st = Statistics(x)
    end = time.time()
    print(f'{end - start:.3g} seconds')
    # 2.83 seconds

Gym
...

`OpenAI Gym
<https://github.com/openai/gym>`_ (or newly `Gymnasium
<https://github.com/Farama-Foundation/Gymnasium>`_)
provides similar functionality as a part of its normalization of observations and rewards
(in gym.wrappers.normalize.RunningMeanStd). The functionality is fairly limited as it was developed for a particular
use case, but the calculation is fast, and it is possible to compose the results. It does not support weights though.
"""

from .version import VERSION as __version__
from .obase import OBase, HandlingInvalid
from .omean import OMean
from .omeanvar import OMeanVar
from .oreg import OReg

__all__ = ['OBase', 'OMean', 'OMeanVar', 'OReg', 'HandlingInvalid']
__author__ = 'Tomas Protivinsky'
