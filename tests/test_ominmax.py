#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import math
from omoment import OMinMax, HandlingInvalid

_inputs_equality = [
    (OMinMax(7, 42, 5), OMinMax(10, 42, 5), False),
    (OMinMax(7, 42, 5), OMinMax(7, 10, 5), False),
    (OMinMax(7, 42, 5), OMinMax(7, 42, 10), False),
    (OMinMax(7, 42, 5), OMinMax(7, 42, 5), True),
]

_inputs_conversion = [
    OMinMax(7, 42, 5),
    OMinMax(0.001, 0.999, 100.),
]

_inputs_addition = [
    (OMinMax(7, 42, 5), OMinMax(0, 10, 10), OMinMax(0, 42, 15)),
    (OMinMax(0, 1, 2), OMinMax(-1, 2, 2), OMinMax(-1, 2, 4)),
    (OMinMax(0, 100, 100), OMinMax(-5, 5, 100), OMinMax(-5, 100, 200)),
]

_inputs_validation = [
    (1, -1, 5, False),
    (1, np.nan, 5, False),
    (np.nan, 10, 5, False),
    (1, 10, -5, False),
    (1, 10, np.nan, False),
    (1, 10, 5, True),
    (0, 0, 1, True),
]

_inputs_getters = [
    (OMinMax(7, 42, 5), 7, 42, 5),
    (OMinMax(0, 100, 500), 0, 100, 500),
]

# the OMeans are mutated during addition
_inputs_combine = [tuple(y.copy() for y in x) for x in _inputs_addition]


@pytest.mark.parametrize('first,second,expected', _inputs_equality)
def test_equality(first, second, expected):
    assert (first == second) == expected


@pytest.mark.parametrize('first,second,expected', _inputs_equality)
def test_close_equality(first, second, expected):
    second.min = second.min + 1e-6
    second.max = second.max - 1e-6
    assert not (first == second), 'Modified OMinMax should not be equal.'
    assert not OMinMax.is_close(first, second), 'Modified OMinMax should not be close enough with default tolerance.'
    assert OMinMax.is_close(first, second, rel_tol=1e-4, abs_tol=1e-4) == expected, 'Modified OMinMax should be ' \
                                                                                    'close enough.'


@pytest.mark.parametrize('input', _inputs_conversion)
def test_conversion(input):
    d = input.to_dict()
    assert (OMinMax.of_dict(d) == input), 'Roundtrip conversion to dict and of dict does not yield the same result.'
    t = input.to_tuple()
    assert (OMinMax.of_tuple(t) == input), 'Roundtrip conversion to tuple and of tuple does not yield the same result.'
    c = input.copy()
    assert (c == input), 'Copy is not equal to original.'
    assert (c is not input), 'Copy is identical object as original.'


@pytest.mark.parametrize('first,second,expected', _inputs_addition)
def test_addition(first, second, expected):
    original_first = first.copy()
    assert OMinMax.is_close(first + second, expected)
    first += second
    assert OMinMax.is_close(first, expected)
    assert not OMinMax.is_close(first, original_first)


@pytest.mark.parametrize('first,second,expected', _inputs_combine)
def test_combine(first, second, expected):
    assert OMinMax.is_close(OMinMax.combine(first, second), expected)
    assert OMinMax.is_close(OMinMax.combine([first, second]), expected)


@pytest.mark.parametrize('min_,max_,n,valid', _inputs_validation)
def test_validation(min_, max_, n, valid):
    if valid:
        OMinMax(min_, max_, n, handling_invalid=HandlingInvalid.Raise)
    else:
        with pytest.raises(ValueError):
            OMinMax(min_, max_, n, handling_invalid=HandlingInvalid.Raise)


@pytest.mark.parametrize('omm,min_,max_,n', _inputs_getters)
def test_get_min_max_n(omm, min_, max_, n):
    assert OMinMax.get_min(omm) == min_
    assert OMinMax.get_max(omm) == max_
    assert OMinMax.get_n(omm) == n


def test_array():
    xarr = np.arange(1, 100)
    assert OMinMax.is_close(OMinMax.compute(xarr), OMinMax(min=1, max=99, n=99))


def test_update():
    xarr = np.arange(20)
    omm = OMinMax(-5, 5, 10)
    omm.update(xarr)
    assert OMinMax.is_close(omm, OMinMax(-5, 19, 30))
    omm.update(xarr)
    assert OMinMax.is_close(omm, OMinMax(-5, 19, 50))


def test_random():
    rng = np.random.Generator(np.random.PCG64(12345))
    xarr = rng.normal(loc=100, scale=20, size=100)
    actual = OMinMax.compute(xarr)
    expected = OMinMax(min=60.9427387397562, max=152.36318852735678, n=100)
    assert OMinMax.is_close(actual, expected)
    assert math.isclose(actual.min, np.min(xarr))
    assert math.isclose(actual.max, np.max(xarr))
    df = pd.DataFrame({'x': xarr})
    actual_of_frame = OMinMax.of_frame(data=df, x='x')
    assert OMinMax.is_close(actual_of_frame, expected)


def test_handling_nans():
    rng = np.random.Generator(np.random.PCG64(54321))
    xarr = rng.normal(loc=100, scale=20, size=100)
    xarr[xarr < 100] = np.nan
    actual = OMinMax.compute(xarr)
    expected = OMinMax(min=100.37912100029071, max=145.7363189455865, n=48)
    assert OMinMax.is_close(actual, expected)
    with pytest.raises(ValueError):
        actual.update(xarr, handling_invalid=HandlingInvalid.Raise)


@pytest.fixture
def df2():
    rng = np.random.Generator(np.random.PCG64(99999))
    n = 1000
    g = rng.integers(low=0, high=10, size=n)
    g2 = rng.integers(low=0, high=2, size=n)
    x = 10 * g + (g2 + 1) * rng.normal(loc=0, scale=50, size=n)
    df = pd.DataFrame({'a': g, 'b': g2, 'c': x})
    return df

def test_of_groupby(df2):
    # no weights
    oms1 = df2.groupby(['a', 'b']).apply(OMinMax.of_frame, x='c')
    oms2 = OMinMax.of_groupby(df2, g=['a', 'b'], x='c')
    for x, y in zip(oms1, oms2):
        assert OMinMax.is_close(x, y)

