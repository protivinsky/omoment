#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
from omoment import OReg, HandlingInvalid


_inputs_equality = [
    (OReg(42, 84, 10, 40, 0, 7), OReg(42, 84, 10, 40, 0, 20), False),
    (OReg(42, 84, 10, 40, 0, 7), OReg(42, 84, 10, 40, 5, 7), False),
    (OReg(42, 84, 10, 40, 0, 7), OReg(42, 84, 10, 10, 0, 7), False),
    (OReg(42, 84, 10, 40, 0, 7), OReg(42, 84, 4, 40, 0, 7), False),
    (OReg(42, 84, 10, 40, 0, 7), OReg(42, 168, 10, 40, 0, 7), False),
    (OReg(42, 84, 10, 40, 0, 7), OReg(0, 84, 10, 40, 0, 7), False),
    (OReg(42, 84, 10, 40, 0, 7), OReg(42, 84, 10, 40, 0, 7), True),
]

_inputs_conversion = [
    OReg(42, 84, 10, 40, 0, 7),
    OReg(4, -4, 16, 9, 1, 10),
]

_inputs_addition = [
    (OReg(0, 0, 5, 20, 5, 10), OReg(0, 0, 15, 60, 15, 10), OReg(0, 0, 10, 40, 10, 20)),
    (OReg(0, 0, 5, 20, -5, 10), OReg(0, 10, 5, 20, -5, 10), OReg(0, 5, 5, 45, -5, 20)),
    (OReg(10, 0, 5, 20, -5, 10), OReg(0, 10, 5, 20, -5, 10), OReg(5, 5, 30, 45, -30, 20)),
]

_inputs_validation = [
    ({'mean_x': 0, 'mean_y': 1, 'var_x': 1, 'var_y': 4, 'cov': 1, 'weight': -1}, False),
    ({'mean_x': 0, 'mean_y': 1, 'var_x': 1, 'var_y': 4, 'cov': 1, 'weight': np.nan}, False),
    ({'mean_x': 0, 'mean_y': 1, 'var_x': 1, 'var_y': np.nan, 'cov': 1, 'weight': 10}, False),
    ({'mean_x': 0, 'mean_y': 1, 'var_x': 1, 'var_y': -1, 'cov': 1, 'weight': 10}, False),
    ({'mean_x': 0, 'mean_y': 1, 'var_x': np.nan, 'var_y': 4, 'cov': 1, 'weight': 10}, False),
    ({'mean_x': 0, 'mean_y': 1, 'var_x': -1, 'var_y': 4, 'cov': 1, 'weight': 10}, False),
    ({'mean_x': 0, 'mean_y': np.nan, 'var_x': 1, 'var_y': 4, 'cov': 1, 'weight': 10}, False),
    ({'mean_x': np.nan, 'mean_y': 1, 'var_x': 1, 'var_y': 4, 'cov': 1, 'weight': 10}, False),
    # test also invalid covariance (greater than sqrt(var_x * var_y)
    ({'mean_x': 0, 'mean_y': 1, 'var_x': 1, 'var_y': 4, 'cov': 3, 'weight': 10}, False),
    ({'mean_x': 0, 'mean_y': 1, 'var_x': 1, 'var_y': 4, 'cov': 1, 'weight': 10}, True),
    ({'mean_x': 0, 'mean_y': 1, 'var_x': 1, 'var_y': 4, 'cov': 1, 'weight': 10}, True),
    # test that weight = 0 is fine with any values, e.g. with default nans
    ({'mean_x': 0}, True),
    ({'mean_x': 0, 'mean_y': 1, 'weight': 10}, False),
]

_inputs_getters = [
    (OReg(42, 84, 10, 40, 0, 7), 42, 84, 10, 40, 0, 7),
    (OReg(4, -4, 16, 9, 1, 10), 4, -4, 16, 9, 1, 10),
]

# the OReg are mutated during addition test
_inputs_combine = [tuple(y.copy() for y in x) for x in _inputs_addition]


@pytest.mark.parametrize('first,second,expected', _inputs_equality)
def test_equality(first, second, expected):
    assert (first == second) == expected


@pytest.mark.parametrize('first,second,expected', _inputs_equality)
def test_close_equality(first, second, expected):
    second.mean_x += 1e-6
    assert not (first == second), 'Modified OReg should not be equal.'
    assert not OReg.is_close(first, second), 'Modified OReg should not be close enough with default tolerance.'
    assert OReg.is_close(first, second, rel_tol=1e-4, abs_tol=1e-4) == expected, 'Modified OReg should be ' \
                                                                                     'close enough.'
    second.mean_y -= 1e-6
    second.var_y += 1e-6
    second.cov += 1e-6
    assert not (first == second), 'Modified OReg should not be equal.'
    assert not OReg.is_close(first, second), 'Modified OReg should not be close enough with default tolerance.'
    assert OReg.is_close(first, second, rel_tol=1e-4, abs_tol=1e-4) == expected, 'Modified OReg should be ' \


@pytest.mark.parametrize('input', _inputs_conversion)
def test_conversion(input):
    d = input.to_dict()
    assert (OReg.of_dict(d) == input), 'Roundtrip conversion to dict and of dict does not yield the same result.'
    t = input.to_tuple()
    assert (OReg.of_tuple(t) == input), 'Roundtrip conversion to tuple and of tuple does not yield the same result.'
    c = input.copy()
    assert (c == input), 'Copy is not equal to original.'
    assert (c is not input), 'Copy is identical object as original.'


@pytest.mark.parametrize('kwargs,valid', _inputs_validation)
def test_validation(kwargs, valid):
    if valid:
        OReg(**kwargs, handling_invalid=HandlingInvalid.Raise)
    else:
        with pytest.raises(ValueError):
            OReg(**kwargs, handling_invalid=HandlingInvalid.Raise)


@pytest.mark.parametrize('first,second,expected', _inputs_addition)
def test_addition(first, second, expected):
    original_first = first.copy()
    assert OReg.is_close(first + second, expected)
    first += second
    assert OReg.is_close(first, expected)
    assert not OReg.is_close(first, original_first)


@pytest.mark.parametrize('first,second,expected', _inputs_combine)
def test_combine(first, second, expected):
    assert OReg.is_close(OReg.combine(first, second), expected)
    assert OReg.is_close(OReg.combine([first, second]), expected)


def test_random1():
    rng = np.random.Generator(np.random.PCG64(12345))
    xarr = rng.normal(loc=100, scale=20, size=100)
    epsarr = rng.normal(loc=0, scale=20, size=100)
    yarr = -50 + 2 * xarr + epsarr
    warr = rng.normal(loc=10, scale=2, size=100)
    actual = OReg.compute(xarr, yarr, warr)
    expected = OReg(mean_x=99.10944975821565, mean_y=149.04818880426825, var_x=348.4832290996651,
                    var_y=1624.4748112150871, cov=659.8255527339259, weight=995.5364462864233)
    assert OReg.is_close(actual, expected)
    df = pd.DataFrame({'x': xarr, 'y': yarr, 'w': warr})
    actual_df = OReg.of_frame(data=df, x='x', y='y', w='w')
    assert OReg.is_close(actual_df, expected)


@pytest.fixture
def df():
    rng = np.random.Generator(np.random.PCG64(99999))
    n = 1000
    g = rng.integers(low=1, high=11, size=n)
    x = 10 * g + rng.normal(loc=0, scale=50, size=n)
    y = -10 * g + rng.normal(loc=0, scale=100, size=n)
    df = pd.DataFrame({'g': g, 'x': x, 'y': y, 'w': g})
    return df


def test_random2(df):
    exp_loc1 = OReg(mean_x=9.929017712742157, mean_y=-10.974025955934188, var_x=2207.1589602628806,
                    var_y=11384.049326235267, cov=-425.8151253703626, weight=94)
    exp_loc10 = OReg(mean_x=96.87023647993738, mean_y=-115.74601703789915, var_x=2126.554880122452,
                     var_y=10075.178149456746, cov=163.85594424149, weight=950)
    exp_overall = OReg(mean_x=69.33027298354544, mean_y=-72.46631413410402, var_x=2994.9736480692286,
                       var_y=11384.94853007606, cov=-509.215400514507, weight=5541)
    ors = df.groupby('g').apply(lambda inner: OReg.of_frame(data=inner, x='x', y='y', w='w'))
    or_overall = OReg.of_frame(df, x='x', y='y', w='w')
    assert OReg.is_close(ors.loc[1], exp_loc1)
    assert OReg.is_close(ors.loc[10], exp_loc10)
    or_combined = OReg.combine(ors)
    assert OReg.is_close(or_overall, exp_overall)
    assert OReg.is_close(or_combined, exp_overall)


@pytest.fixture
def df2():
    rng = np.random.Generator(np.random.PCG64(99999))
    n = 1000
    g = rng.integers(low=0, high=10, size=n)
    g2 = rng.integers(low=0, high=2, size=n)
    x = 10 * g + (g2 + 1) * rng.normal(loc=0, scale=50, size=n)
    y = -10 * g + (g2 - 10) * rng.normal(loc=0, scale=100, size=n)
    w = rng.exponential(scale=1, size=n)
    df = pd.DataFrame({'a': g, 'b': g2, 'c': x, 'd': y, 'e': w})
    return df


def test_results(df, df2):
    df2_renamed = df2.rename(columns={'a': 'g', 'b': 'g2', 'c': 'x', 'd': 'y', 'e': 'w'})
    for ff in [df, df2_renamed]:
        ors = ff.groupby('g').apply(OReg.of_frame, x='x', y='y', w='w')
        or_overall = OReg.combine(ors)
        np_mean_x = np.average(ff['x'], weights=ff['w'])
        np_mean_y = np.average(ff['y'], weights=ff['w'])
        assert math.isclose(or_overall.mean_x, np_mean_x)
        assert math.isclose(or_overall.mean_y, np_mean_y)
        ors2 = ff.groupby('g').apply(OReg.of_frame, x='x', y='y')
        or2_overall = OReg.combine(ors2)
        np_std_dev_x = np.std(ff['x'])
        np_std_dev_y = np.std(ff['y'])
        np_cov_matrix = np.cov(ff['x'], ff['y'], ddof=0)
        assert math.isclose(or2_overall.std_dev_x, np_std_dev_x)
        assert math.isclose(or2_overall.std_dev_y, np_std_dev_y)
        assert math.isclose(or2_overall.var_x, np_cov_matrix[0, 0])
        assert math.isclose(or2_overall.var_y, np_cov_matrix[1, 1])
        assert math.isclose(or2_overall.cov, np_cov_matrix[1, 0])
        np_corr = np.corrcoef(ff['x'], ff['y'])[1, 0]
        assert math.isclose(or2_overall.corr, np_corr)


def test_handling_nans():
    rng = np.random.Generator(np.random.PCG64(54321))
    xarr = rng.normal(loc=100, scale=20, size=100)
    epsarr = rng.normal(loc=0, scale=20, size=100)
    yarr = -50 + 2 * xarr + epsarr
    warr = rng.normal(loc=10, scale=2, size=100)
    xarr[xarr < 100] = np.nan
    yarr[yarr > 160] = np.nan
    warr[warr < 10] = np.nan
    actual = OReg.compute(xarr, yarr, warr)
    expected = OReg(mean_x=108.19676769040585, mean_y=147.33514065030363, var_x=17.21726414913506,
                    var_y=127.50989780754267, cov=-6.144415084260051, weight=45.24780957888226)
    assert OReg.is_close(actual, expected)
    with pytest.raises(ValueError):
        actual.update(xarr, yarr, warr, handling_invalid=HandlingInvalid.Raise)


@pytest.mark.parametrize('oreg,mean_x,mean_y,var_x,var_y,cov,weight', _inputs_getters)
def test_getters(oreg, mean_x, mean_y, var_x, var_y, cov, weight):
    assert OReg.get_mean_x(oreg) == mean_x
    assert OReg.get_mean_y(oreg) == mean_y
    assert OReg.get_var_x(oreg) == var_x
    assert OReg.get_var_y(oreg) == var_y
    assert OReg.get_cov(oreg) == cov
    assert OReg.get_weight(oreg) == weight
    assert OReg.get_mean_x(oreg) == oreg.mean_x
    assert OReg.get_mean_y(oreg) == oreg.mean_y
    assert OReg.get_var_x(oreg) == oreg.var_x
    assert OReg.get_var_y(oreg) == oreg.var_y
    assert OReg.get_cov(oreg) == oreg.cov
    assert OReg.get_weight(oreg) == oreg.weight
    assert OReg.get_std_dev_x(oreg) == oreg.std_dev_x
    assert OReg.get_std_dev_y(oreg) == oreg.std_dev_y
    assert OReg.get_corr(oreg) == oreg.corr
    assert OReg.get_alpha(oreg) == oreg.alpha
    assert OReg.get_beta(oreg) == oreg.beta


def test_of_groupby(df2):
    # no weights
    ors1 = df2.groupby(['a', 'b']).apply(OReg.of_frame, x='c', y='d')
    ors2 = OReg.of_groupby(df2, g=['a', 'b'], x='c', y='d')
    for x, y in zip(ors1, ors2):
        assert OReg.is_close(x, y)

    # with weights
    ors1 = df2.groupby(['a', 'b']).apply(OReg.of_frame, x='c', y='d', w='e')
    ors2 = OReg.of_groupby(df2, g=['a', 'b'], x='c', y='d', w='e')
    for x, y in zip(ors1, ors2):
        assert OReg.is_close(x, y)


def test_regression(df):
    res = sm.WLS(df['y'], sm.add_constant(df['x']), weights=df['w']).fit()
    oreg = OReg.of_frame(df, x='x', y='y', w='w')
    assert math.isclose(oreg.alpha, res.params['const'])
    assert math.isclose(oreg.beta, res.params['x'])

