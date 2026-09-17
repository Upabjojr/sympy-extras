"""Tests of Trager's algorithm for algebraic functions (quadratic extensions)."""
from __future__ import annotations

from sympy import symbols, sqrt, log, I, S, simplify, exp, sin
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import as_expr
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
    # two roots: the product of the radicands (test_several_roots_are_combined_and_restored)
    product = trager_antiderivative(sqrt(x) * sqrt(x + 1), x)
    assert product is not None and product.has(sqrt(x) * sqrt(x + 1))
    assert trager_antiderivative(sqrt(x) + sqrt(x + 1), x) is None
    assert trager_antiderivative(sin(x), x) is None
    assert hermite_reduce_algebraic(sqrt(sqrt(x) + 1), x) is None
    # a parameter: the Hermite reduction only
    found = hermite_reduce_algebraic(x**2 / sqrt(x**2 + a), x)
    assert found is not None and simplify(found[0].diff(x) + found[1] - x**2 / sqrt(x**2 + a)) == 0
    assert is_nonelementary_algebraic(1 / sqrt(x**2 + a), x) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(trager_antiderivative)([1], x))
    raises(TypeError, lambda: untyped(trager_reduce)(None, x))


def test_perfect_square_radicands_carry_the_sign() -> None:
    # sqrt(d**2) is d*sign(d): the antiderivative of the rational
    # integrand on each component, and the verification ignores the jump
    from sympy import sign
    from sympy_extras.integrals import verified_antiderivative
    b, c = symbols('b c')
    assert trager_antiderivative(sqrt(x**2), x) == x**2 * sign(x) / 2
    assert trager_antiderivative(sqrt(x**2 + 2 * x + 1) / (x + 3), x) == (x - 2 * log(x + 3)) * sign(x + 1)
    f = as_expr((b**2 / (4 * c) + b * x + c * x**2)**(-S(3) / 2))
    found = verified_antiderivative(f, x)
    assert found is not None and found[0].has(sign(b / (2 * c) + x))


def test_several_roots_are_combined_and_restored() -> None:
    # sqrt(x + 1)*sqrt(x + 2) is the root of the product where both are
    # positive and its negative below -2: the algebra runs in the root of
    # the product, the answer is written in the original roots and holds
    # on both sides
    from sympy_extras.integrals import is_antiderivative, verified_antiderivative
    f = sqrt(x + 1) * sqrt(x + 2) / x
    F = trager_antiderivative(f, x)
    assert F is not None and F.has(sqrt(x + 1) * sqrt(x + 2)) and not F.has(sqrt(x**2 + 3 * x + 2))
    assert is_antiderivative(F, f, x) is True
    found = verified_antiderivative(f, x)
    assert found is not None and found[1] == 'trager'
    # a monomial with one root only is not rational in the product
    assert trager_antiderivative(sqrt(x + 1) * sqrt(x + 2) + sqrt(x + 1), x) is None
    # the product of the radicands of genus one: the elliptic remainder
    found_ = trager_reduce((x**2 - 1) / (x**2 * sqrt((1 - 3 * x**2) * (1 - x**2))), x)
    assert found_ is not None and found_[1] != 0
