#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import math
from omoment import OMean, HandlingInvalid

_inputs_equality = [
    (OMean(42, 7), OMean(42, 10), False),
    (OMean(42, 7), OMean(7, 7), False),
    (OMean(42, 7), OMean(42, 7), True),
]

_inputs_conversion = [
    OMean(42, 7),
    OMean(0, 100),
]

_inputs_addition = [
    (OMean(2, 1), OMean(10, 1), OMean(6, 2)),
    (OMean(1, 1), OMean(5, 3), OMean(4, 4)),
    (OMean(0, 100), OMean(100, 100), OMean(50, 200)),
]

_inputs_validation = [
    (1, -1, False),
    (5, np.inf, False),
    (np.nan, 10, False),
    (np.inf, 0, True),
    (0, 0, True),
    (10, 0, True),
    (0, 1, True),
]

_inputs_getters = [
    (OMean(42, 7), 42, 7),
    (OMean(0, 100), 0, 100),
]

# the OMeans are mutated during addition
_inputs_combine = [tuple(y.copy() for y in x) for x in _inputs_addition]


@pytest.mark.parametrize('first,second,expected', _inputs_equality)
def test_equality(first, second, expected):
    assert (first == second) == expected


@pytest.mark.parametrize('first,second,expected', _inputs_equality)
def test_close_equality(first, second, expected):
    second.mean = second.mean + 1e-6
    assert not (first == second), 'Modified OMean should not be equal.'
    assert not OMean.is_close(first, second), 'Modified OMean should not be close enough with default tolerance.'
    assert OMean.is_close(first, second, rel_tol=1e-4, abs_tol=1e-4) == expected, 'Modified OMean should be close ' \
                                                                                  'enough.'


@pytest.mark.parametrize('input', _inputs_conversion)
def test_conversion(input):
    d = input.to_dict()
    assert (OMean.of_dict(d) == input), 'Roundtrip conversion to dict and of dict does not yield the same result.'
    t = input.to_tuple()
    assert (OMean.of_tuple(t) == input), 'Roundtrip conversion to tuple and of tuple does not yield the same result.'
    c = input.copy()
    assert (c == input), 'Copy is not equal to original.'
    assert (c is not input), 'Copy is identical object as original.'


@pytest.mark.parametrize('first,second,expected', _inputs_addition)
def test_addition(first, second, expected):
    original_first = first.copy()
    assert OMean.is_close(first + second, expected)
    first += second
    assert OMean.is_close(first, expected)
    assert not OMean.is_close(first, original_first)


@pytest.mark.parametrize('first,second,expected', _inputs_combine)
def test_combine(first, second, expected):
    assert OMean.is_close(OMean.combine(first, second), expected)
    assert OMean.is_close(OMean.combine([first, second]), expected)


@pytest.mark.parametrize('mean,weight,valid', _inputs_validation)
def test_validation(mean, weight, valid):
    if valid:
        OMean(mean, weight, handling_invalid=HandlingInvalid.Raise)
    else:
        with pytest.raises(ValueError):
            OMean(mean, weight, handling_invalid=HandlingInvalid.Raise)


def test_update():
    xarr = np.full(10, fill_value=10)
    warr = np.full(10, fill_value=5)
    om = OMean(0, 40)
    om.update(xarr)
    assert OMean.is_close(om, OMean(2, 50))
    om.update(xarr, warr)
    assert OMean.is_close(om, OMean(6, 100))


def test_array():
    xarr = np.arange(1, 100)
    warr = np.arange(1, 100)
    assert OMean.is_close(OMean.compute(xarr, warr), OMean(mean=66.33333333333333, weight=4950))


def test_random():
    rng = np.random.Generator(np.random.PCG64(12345))
    xarr = rng.normal(loc=100, scale=20, size=100)
    warr = rng.normal(loc=10, scale=2, size=100)
    actual = OMean.compute(xarr, warr)
    expected = OMean(mean=99.08210157394731, weight=1006.9717003477731)
    assert OMean.is_close(actual, expected)
    assert math.isclose(actual.mean, np.average(xarr, weights=warr))
    assert math.isclose(actual.weight, np.sum(warr))
    df = pd.DataFrame({'x': xarr, 'w': warr})
    actual_of_frame = OMean.of_frame(data=df, x='x', w='w')
    assert OMean.is_close(actual_of_frame, expected)


def test_handling_nans():
    rng = np.random.Generator(np.random.PCG64(54321))
    xarr = rng.normal(loc=100, scale=20, size=100)
    warr = rng.normal(loc=10, scale=2, size=100)
    xarr[xarr < 100] = np.nan
    warr[warr < 10] = np.nan
    actual = OMean.compute(xarr, warr)
    expected = OMean(mean=113.75735031907175, weight=272.7794894778689)
    assert OMean.is_close(actual, expected)
    with pytest.raises(ValueError):
        actual.update(xarr, warr, handling_invalid=HandlingInvalid.Raise)


@pytest.mark.parametrize('om,mean,weight', _inputs_getters)
def test_get_mean_weight(om, mean, weight):
    assert OMean.get_mean(om) == mean
    assert OMean.get_weight(om) == weight


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
    oms1 = df2.groupby(['a', 'b']).apply(OMean.of_frame, x='c')
    oms2 = OMean.of_groupby(df2, g=['a', 'b'], x='c')
    for x, y in zip(oms1, oms2):
        assert OMean.is_close(x, y)

    # weights
    oms1 = df2.groupby(['a', 'b']).apply(OMean.of_frame, x='c', w='d')
    oms2 = OMean.of_groupby(df2, g=['a', 'b'], x='c', w='d')
    for x, y in zip(oms1, oms2):
        assert OMean.is_close(x, y)

