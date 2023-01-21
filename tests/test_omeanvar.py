#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import math
from omoment import OMeanVar, HandlingInvalid

_inputs_equality = [
    (OMeanVar(42, 10, 7), OMeanVar(42, 10, 20), False),
    (OMeanVar(42, 10, 7), OMeanVar(42, 7, 7), False),
    (OMeanVar(42, 10, 7), OMeanVar(7, 10, 7), False),
    (OMeanVar(42, 10, 7), OMeanVar(42, 10, 7), True),
]

_inputs_conversion = [
    OMeanVar(42, 7, 10),
    OMeanVar(0, 1, 100),
]

_inputs_addition = [
    (OMeanVar(0, 5, 10), OMeanVar(0, 15, 10), OMeanVar(0, 10, 20)),
    (OMeanVar(0, 5, 10), OMeanVar(10, 5, 10), OMeanVar(5, 30, 20)),
]

_inputs_validation = [
    ({'mean': 0, 'var': 1, 'weight': -1}, False),
    ({'mean': 1, 'var': 1, 'weight': np.nan}, False),
    ({'mean': 5, 'var': np.inf, 'weight': 10}, False),
    ({'mean': 4, 'var': -1, 'weight': 10}, False),
    ({'mean': np.inf, 'var': 10, 'weight': 10}, False),
    ({'mean': 0, 'var': 0, 'weight': 1}, True),
    ({'mean': 10, 'var': 0, 'weight': 10}, True),
    ({'mean': 0, 'var': 1, 'weight': 1}, True),
    ({'mean': np.inf, 'var': np.inf, 'weight': 0}, True),
    ({'mean': 1}, True),
    ({'mean': 1, 'weight': 1}, False),
]

_inputs_getters = [
    (OMeanVar(42, 16, 10), 42, 16, 10),
    (OMeanVar(0, 1, 100), 0, 1, 100),
]

# the OMeanVars are mutated during addition
_inputs_combine = [tuple(y.copy() for y in x) for x in _inputs_addition]


@pytest.mark.parametrize('first,second,expected', _inputs_equality)
def test_equality(first, second, expected):
    assert (first == second) == expected


@pytest.mark.parametrize('first,second,expected', _inputs_equality)
def test_close_equality(first, second, expected):
    second.mean += 1e-6
    assert not (first == second), 'Modified OMeanVar should not be equal.'
    assert not OMeanVar.is_close(first, second), 'Modified OMeanVar should not be close enough with default tolerance.'
    assert OMeanVar.is_close(first, second, rel_tol=1e-4, abs_tol=1e-4) == expected, 'Modified OMeanVar should be ' \
                                                                                     'close enough.'
    second.mean -= 1e-6
    second.var += 1e-6
    assert not (first == second), 'Modified OMeanVar should not be equal.'
    assert not OMeanVar.is_close(first, second), 'Modified OMeanVar should not be close enough with default tolerance.'
    assert OMeanVar.is_close(first, second, rel_tol=1e-4, abs_tol=1e-4) == expected, 'Modified OMeanVar should be ' \
                                                                                     'close enough.'


@pytest.mark.parametrize('input', _inputs_conversion)
def test_conversion(input):
    d = input.to_dict()
    assert (OMeanVar.of_dict(d) == input), 'Roundtrip conversion to dict and of dict does not yield the same result.'
    t = input.to_tuple()
    assert (OMeanVar.of_tuple(t) == input), 'Roundtrip conversion to tuple and of tuple does not yield the same result.'
    c = input.copy()
    assert (c == input), 'Copy is not equal to original.'
    assert (c is not input), 'Copy is identical object as original.'


@pytest.mark.parametrize('kwargs,valid', _inputs_validation)
def test_validation(kwargs, valid):
    if valid:
        OMeanVar(**kwargs, handling_invalid=HandlingInvalid.Raise)
    else:
        with pytest.raises(ValueError):
            OMeanVar(**kwargs, handling_invalid=HandlingInvalid.Raise)


@pytest.mark.parametrize('first,second,expected', _inputs_addition)
def test_addition(first, second, expected):
    original_first = first.copy()
    assert OMeanVar.is_close(first + second, expected)
    first += second
    assert OMeanVar.is_close(first, expected)
    assert not OMeanVar.is_close(first, original_first)


@pytest.mark.parametrize('first,second,expected', _inputs_combine)
def test_combine(first, second, expected):
    assert OMeanVar.is_close(OMeanVar.combine(first, second), expected)
    assert OMeanVar.is_close(OMeanVar.combine([first, second]), expected)


def test_random1():
    rng = np.random.Generator(np.random.PCG64(12345))
    xarr = rng.normal(loc=100, scale=20, size=100)
    warr = rng.normal(loc=10, scale=2, size=100)
    actual = OMeanVar.compute(xarr, warr)
    expected = OMeanVar(mean=99.08210157394731, var=344.8769032099984, weight=1006.9717003477731)
    assert OMeanVar.is_close(actual, expected)
    df = pd.DataFrame({'x': xarr, 'w': warr})
    actual_df = OMeanVar.of_frame(data=df, x='x', w='w')
    assert OMeanVar.is_close(actual_df, expected)


@pytest.fixture
def df():
    rng = np.random.Generator(np.random.PCG64(99999))
    n = 1000
    g = rng.integers(low=1, high=11, size=n)
    x = 10 * g + rng.normal(loc=0, scale=50, size=n)
    df = pd.DataFrame({'g': g, 'x': x, 'w': g})
    return df


def test_random2(df):
    exp_loc1 = OMeanVar(mean=9.929017712742157, var=2207.1589602628806, weight=94)
    exp_loc10 = OMeanVar(mean=96.87023647993738, var=2126.554880122452, weight=950)
    exp_overall = OMeanVar(mean=69.33027298354546, var=2994.9736480692286, weight=5541)
    omvs = df.groupby('g').apply(lambda inner: OMeanVar.of_frame(data=inner, x='x', w='w'))
    assert OMeanVar.is_close(omvs.loc[1], exp_loc1)
    assert OMeanVar.is_close(omvs.loc[10], exp_loc10)
    omv_overall = OMeanVar.combine(omvs)
    assert OMeanVar.is_close(omv_overall, exp_overall)


def test_results(df):
    omvs = df.groupby('g').apply(OMeanVar.of_frame, x='x', w='w')
    omv_overall = OMeanVar.combine(omvs)
    np_mean = np.average(df['x'], weights=df['w'])
    assert math.isclose(omv_overall.mean, np_mean)
    omvs2 = df.groupby('g').apply(OMeanVar.of_frame, x='x')
    omv2_overall = OMeanVar.combine(omvs2)
    np_std_dev = np.std(df['x'])
    assert math.isclose(omv2_overall.std_dev, np_std_dev)


def test_handling_nans():
    rng = np.random.Generator(np.random.PCG64(54321))
    xarr = rng.normal(loc=100, scale=20, size=100)
    warr = rng.normal(loc=10, scale=2, size=100)
    xarr[xarr < 100] = np.nan
    warr[warr < 10] = np.nan
    actual = OMeanVar.compute(xarr, warr)
    expected = OMeanVar(mean=113.75735031907175, var=130.74635001488218, weight=272.7794894778689)
    assert OMeanVar.is_close(actual, expected)
    with pytest.raises(ValueError):
        actual.update(xarr, warr, handling_invalid=HandlingInvalid.Raise)


@pytest.mark.parametrize('om,mean,var,weight', _inputs_getters)
def test_get_mean_weight_var(om, mean, var, weight):
    assert OMeanVar.get_mean(om) == mean
    assert OMeanVar.get_var(om) == var
    assert OMeanVar.get_weight(om) == weight
    assert OMeanVar.get_std_dev(om) == om.std_dev
    assert OMeanVar.get_unbiased_var(om) == om.unbiased_var
    assert OMeanVar.get_unbiased_std_dev(om) == om.unbiased_std_dev


@pytest.fixture
def df2():
    rng = np.random.Generator(np.random.PCG64(99999))
    n = 1000
    g = rng.integers(low=0, high=10, size=n)
    g2 = rng.integers(low=0, high=2, size=n)
    x = 10 * g + (g2 + 1) * rng.normal(loc=0, scale=50, size=n)
    w = rng.exponential(scale=1, size=n)
    df = pd.DataFrame({'a': g, 'b': g2, 'c': x, 'd': w})
    return df


def test_of_groupby(df2):
    # no weights
    omvs1 = df2.groupby(['a', 'b']).apply(OMeanVar.of_frame, x='c')
    omvs2 = OMeanVar.of_groupby(df2, g=['a', 'b'], x='c')
    for x, y in zip(omvs1, omvs2):
        assert OMeanVar.is_close(x, y)

    # with weights
    omvs1 = df2.groupby(['a', 'b']).apply(OMeanVar.of_frame, x='c', w='d')
    omvs2 = OMeanVar.of_groupby(df2, g=['a', 'b'], x='c', w='d')
    for x, y in zip(omvs1, omvs2):
        assert OMeanVar.is_close(x, y)

