"""Tests of Trager's algorithm for algebraic functions (quadratic extensions)."""
from __future__ import annotations

from sympy import symbols, sqrt, log, I, simplify, exp, sin
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.integrals.trager import (trager_antiderivative, trager_reduce, hermite_reduce_algebraic,
                                           is_nonelementary_algebraic)

x, a = symbols('x a')


def _derivative_matches(F: object, f: object) -> bool:
    from sympy_extras._typing import as_expr
    return simplify(as_expr(F).diff(x) - as_expr(f)) == 0


def test_antiderivatives_are_verified_by_differentiation() -> None:
    cases = [
        (1 / (x * sqrt(x**2 + 1)), -log((sqrt(x**2 + 1) + 1) / x)),
        (sqrt(x**2 + 1) / x, sqrt(x**2 + 1) - log((sqrt(x**2 + 1) + 1) / x)),
        (x / sqrt(x**4 + 1), log(x**2 + sqrt(x**4 + 1)) / 2),
        (1 / sqrt(x**2 - 1), log(x + sqrt(x**2 - 1))),
        (x**2 / sqrt(x**2 + 1), x * sqrt(x**2 + 1) / 2 - log(x + sqrt(x**2 + 1)) / 2),
        (1 / (x * sqrt(x**3 + 1)), -log((x**3 / 2 + sqrt(x**3 + 1) + 1) / x**3) / 3),
        (1 / (x * sqrt(x**4 + 1)), -log((sqrt(x**4 + 1) + 1) / x**2) / 2),
    ]
    for f, expected in cases:
        found = trager_antiderivative(f, x)
        assert found is not None and found == expected, (f, found)
        assert _derivative_matches(found, f)
    # a class of residues in a number field: the logarithm of a function
    # with complex coefficients (a real form would be an arctangent)
    found = trager_antiderivative((x**2 - 1) / ((x**2 + 1) * sqrt(x**4 + 1)), x)
    assert found is not None and found.has(I) and _derivative_matches(found, (x**2 - 1) / ((x**2 + 1) * sqrt(x**4 + 1)))
    found = trager_antiderivative(1 / ((x**2 + 1) * sqrt(x**2 + 2)), x)
    assert found is not None and _derivative_matches(found, 1 / ((x**2 + 1) * sqrt(x**2 + 2)))


def test_hermite_reduction() -> None:
    # multiple poles removed, the polynomial part reduced at infinity
    found = hermite_reduce_algebraic(x**2 / sqrt(x**2 + 1), x)
    assert found == (x * sqrt(x**2 + 1) / 2, -1 / (2 * sqrt(x**2 + 1)))
    found = hermite_reduce_algebraic(1 / (x**2 * sqrt(x**3 + 1)), x)
    assert found == (-sqrt(x**3 + 1) / x, x / (2 * sqrt(x**3 + 1)))
    f = (x**3 + 1) / ((x - 1)**2 * sqrt(x**3 + 1))
    found = hermite_reduce_algebraic(f, x)
    assert found is not None
    g, h = found
    assert simplify(g.diff(x) + h - f) == 0
    # a ramified pole (a factor of the radicand in the denominator)
    f = 1 / (x * sqrt(x)) + 1 / sqrt(x)
    found = hermite_reduce_algebraic(f, x)
    assert found is not None and simplify(found[0].diff(x) + found[1] - f) == 0
    # a rational part is reduced too (Horowitz-Ostrogradsky)
    f = 1 / (x - 1)**2 + 1 / sqrt(x**2 + 1)
    found = hermite_reduce_algebraic(f, x)
    assert found is not None and simplify(found[0].diff(x) + found[1] - f) == 0 and found[0].has(1 / (x - 1))


def test_nonelementary_integrals_are_left_alone() -> None:
    # differentials of the first kind and poles of order two at infinity
    assert trager_antiderivative(1 / sqrt(x**3 + 1), x) is None
    assert trager_reduce(1 / sqrt(x**3 + 1), x) == (0, 1 / sqrt(x**3 + 1))
    assert trager_reduce(1 / sqrt(1 - x**4), x) == (0, 1 / sqrt(1 - x**4))
    assert is_nonelementary_algebraic(x / sqrt(x**3 + 1), x) is True
    assert is_nonelementary_algebraic(1 / (x**2 * sqrt(x**3 + 1)), x) is True
    assert is_nonelementary_algebraic(x / sqrt(x**4 + 1), x) is False
    # the elementary part split off, the rest a differential of the first kind
    f = (2 * x**2 + 1) / ((x**2 + 1) * sqrt(x**4 + x**2 + 1))
    found = trager_reduce(f, x)
    assert found is not None
    F, h = found
    assert h == 3 / (2 * sqrt(x**4 + x**2 + 1)) and simplify(F.diff(x) + h - f) == 0
    assert is_nonelementary_algebraic(f, x) is True
    # an irrational residue on a curve of genus one: not decided
    assert is_nonelementary_algebraic(1 / ((x - 1) * sqrt(x**3 + 1)), x) is None


def test_integrands_outside_the_scope() -> None:
    assert trager_antiderivative(exp(x) * sqrt(x), x) is None
    assert trager_antiderivative(sqrt(x) * sqrt(x + 1), x) is None
    assert trager_antiderivative(sin(x), x) is None
    assert hermite_reduce_algebraic(sqrt(sqrt(x) + 1), x) is None
    # a parameter: the Hermite reduction only
    found = hermite_reduce_algebraic(x**2 / sqrt(x**2 + a), x)
    assert found is not None and simplify(found[0].diff(x) + found[1] - x**2 / sqrt(x**2 + a)) == 0
    assert is_nonelementary_algebraic(1 / sqrt(x**2 + a), x) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(trager_antiderivative)([1], x))
    raises(TypeError, lambda: untyped(trager_reduce)(None, x))
