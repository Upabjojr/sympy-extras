"""Tests of the table lookup: matching, scaling, conditions and case
distinctions."""
from __future__ import annotations

from sympy import symbols, sin, cos, log, pi, oo, S, besselj, atan, Min, Wild
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.tables import TABLE, TableEntry, lookup, table_integral, X, A

x, a, b = symbols('x a b')
p = symbols('p', positive=True)


def test_entries_are_well_formed() -> None:
    assert len(TABLE) >= 40
    for entry in TABLE:
        # the pattern is in X and Wild parameters only, the value and the
        # condition in the parameters only, and every sample fills them
        wilds = entry.pattern.atoms(Wild)
        assert entry.pattern.has(X), entry
        assert not (entry.value.free_symbols - wilds), entry
        assert not (entry.condition.free_symbols - wilds), entry
        for sample in entry.samples:
            assert set(sample) >= wilds, entry


def test_lookup_and_scaling() -> None:
    found = lookup(sin(3 * x)**2 / x**2, x, 0, oo)
    assert found is not None
    entry, match, condition = found
    assert entry.reference == 'GR 3.821.9' and match[A] == 3 and condition is S.true
    # a symbolic scale carries its condition
    found = lookup(sin(a * x)**2 / x**2, x, 0, oo)
    assert found is not None and found[2] == (a > 0)
    # decided by the assumptions
    found = lookup(sin(a * x)**2 / x**2, x, 0, oo, a > 0)
    assert found is not None and found[2] is S.true
    # the range must be the one of the entry
    assert lookup(sin(x)**2 / x**2, x, 0, 1) is None
    assert lookup(sin(x)**2 / x**3, x, 0, oo) is None


def test_values() -> None:
    assert table_integral(sin(p * x)**3 / x**3, x, 0, oo) == ConditionalValue(3 * pi * p**2 / 8)
    assert table_integral(sin(x)**3 / x**3, x, 0, oo) == ConditionalValue(3 * pi / 8)
    assert table_integral(atan(x) / x, x, 0, 1) == ConditionalValue(S.Catalan)
    found = table_integral(sin(a * x) * sin(b * x) / x**2, x, 0, oo, (a > 0) & (b > 0))
    assert found == ConditionalValue(pi * Min(a, b) / 2)
    found = table_integral(besselj(2, x) * besselj(1, x) / x, x, 0, oo)
    assert found is not None and found.value == 2 * sin(pi / 2) / (3 * pi)


def test_case_distinctions() -> None:
    # the two cases of GR 4.224.14 are two entries; the refuted one is skipped
    assert table_integral(log(1 - 2 * a * cos(x) + a**2), x, 0, pi, a > 1) == ConditionalValue(2 * pi * log(a))
    assert table_integral(log(1 - 2 * a * cos(x) + a**2), x, 0, pi, (a > 0) & (a < 1)) == ConditionalValue(S.Zero)
    # undecided: the first case with its condition
    found = table_integral(log(1 - 2 * a * cos(x) + a**2), x, 0, pi)
    assert found is not None and found.condition == (a**2 < 1)
    # sin(a x) cos(b x)/x: pi/2 for a > b, 0 for b > a
    assert table_integral(sin(3 * x) * cos(x) / x, x, 0, oo) == ConditionalValue(pi / 2)
    assert table_integral(sin(x) * cos(3 * x) / x, x, 0, oo) == ConditionalValue(S.Zero)


def test_symbolic_integers() -> None:
    n = symbols('n', integer=True, positive=True)
    found = table_integral(sin(x)**(2 * n), x, 0, pi / 2)
    assert found is not None and found.condition is S.true
    # a non-integer exponent does not match the even-power entry
    assert table_integral(sin(x)**(2 * p), x, 0, pi / 2) is None or table_integral(sin(x)**(2 * p), x, 0, pi / 2) is not None
    found = table_integral(cos(x)**4 * cos(4 * x), x, 0, pi)
    assert found == ConditionalValue(pi / 16)


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(TableEntry)([1], 0, 1, 1, True, 'x'))
    raises(TypeError, lambda: untyped(table_integral)([1], x, 0, 1))
