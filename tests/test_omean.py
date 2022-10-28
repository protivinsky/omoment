import pytest
from omoment import OMean

_inputs_equality = ([
    (OMean(42, 7), OMean(42, 10), False),
    (OMean(42, 7), OMean(7, 7), False),
    (OMean(42, 7), OMean(42, 7), True),
])

_inputs_conversion = ([
    OMean(42, 7),
    OMean(0, 100),
])

_inputs_addition = ([
    (OMean(2, 1), OMean(10, 1), OMean(6, 2)),
    (OMean(1, 1), OMean(5, 3), OMean(4, 4)),
    (OMean(0, 100), OMean(100, 100), OMean(50, 200)),
])


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

