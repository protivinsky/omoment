import pytest
import numpy as np
import pandas as pd
import math
from omoment import OMean

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

_inputs_validation = [(1, -1), (5, np.inf), (np.nan, 10)]

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


@pytest.mark.parametrize('mean,weight', _inputs_validation)
def test_validation(mean, weight):
    with pytest.raises(ValueError):
        OMean(mean, weight)


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
    assert OMean.is_close(OMean(xarr, warr), OMean(mean=66.33333333333333, weight=4950))
    om = OMean()
    om.update(xarr, warr)
    assert OMean.is_close(om, OMean(mean=66.33333333333333, weight=4950))


def test_random():
    rng = np.random.Generator(np.random.PCG64(12345))
    xarr = rng.normal(loc=100, scale=20, size=100)
    warr = rng.normal(loc=10, scale=2, size=100)
    actual = OMean(xarr, warr)
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
    actual = OMean(xarr, warr)
    expected = OMean(mean=113.75735031907175, weight=272.7794894778689)
    assert OMean.is_close(actual, expected)
    with pytest.raises(ValueError):
        actual.update(xarr, warr, raise_if_nans=True)



