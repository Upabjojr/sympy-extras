"""Tests of Ramanujan's master theorem and the method of brackets."""
from __future__ import annotations

from sympy import symbols, exp, sin, cos, log, atan, erf, sqrt, oo, gamma, pi, S, Rational, besselj
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals.brackets import (taylor_coefficient, ramanujan_master_theorem, method_of_brackets,
                                             mellin_transform_series)
from sympy_extras.integrals.marichev import mellin_integrate
from sympy_extras.integrals.mellin import mellin_transform

x, s = symbols('x s')


def _agree(u: ExprLike, v: ExprLike, point: Rational) -> bool:
    return abs(complex(as_expr(u).subs(s, point).evalf(20)) - complex(as_expr(v).subs(s, point).evalf(20))) < 1e-12


def test_taylor_coefficient() -> None:
    c = taylor_coefficient(exp(-x), x)
    assert c is not None and (c.period, c.offset) == (1, 0)
    assert [c.coefficient.subs(c.index, k) for k in range(4)] == [1, -1, S.Half, -Rational(1, 6)]
    c = taylor_coefficient(sin(x), x)
    assert c is not None and (c.period, c.offset) == (2, 1)
    assert [c.coefficient.subs(c.index, k) for k in range(3)] == [1, -Rational(1, 6), Rational(1, 120)]
    c = taylor_coefficient(log(1 + x), x)
    assert c is not None and c.coefficient.subs(c.index, 2) == -S.Half
    # phi(j) = (-1)^j j! c(j) of the master theorem
    phi = c.phi()
    assert phi is not None and phi.subs(c.index, 3) == -2
    # no single closed form (two residue classes), and no series at all
    assert taylor_coefficient(besselj(0, x), x) is None
    c = taylor_coefficient(exp(x), x)
    assert c is not None and c.phi() is None


def test_master_theorem_reproduces_the_table() -> None:
    # the two independent derivations must agree: the table (from the
    # literature) and the theorem (from the power series)
    cases = [(exp(-x), 'exp', Rational(3, 2)), (1 / (1 + x), '(1 + x)**(-a)', S.Half), (sin(x), 'sin', S.Half),
             (cos(x), 'cos', S.Half), (erf(x), 'erf', -S.Half), (log(1 + x), 'log(1 + x)', -S.Half),
             (atan(x), 'atan', -S.Half)]
    for f, name, point in cases:
        found = ramanujan_master_theorem(f, x, s)
        assert found is not None, name
        table = mellin_transform(f, x, s)
        assert table is not None
        assert _agree(found.transform, table.transform, point), name
        assert found.strip == table.strip, name


def test_master_theorem_strips_and_failures() -> None:
    found = ramanujan_master_theorem(exp(-x**2), x, s)
    assert found is not None and found.strip == (0, oo) and found.transform == gamma(s / 2) / 2
    found = ramanujan_master_theorem(cos(sqrt(x)), x, s)
    assert found is not None and found.strip == (0, S.Half)
    # M[f(sqrt(x))](s) = 2 M[f](2 s)
    assert _agree(found.transform, 2 * gamma(2 * s) * cos(pi * s), Rational(1, 3))
    # exp(x): the coefficients keep (-1)^k, no transform
    assert ramanujan_master_theorem(exp(x), x, s) is None
    assert ramanujan_master_theorem(x**2 + 1, x, s) is None


def test_scaled_and_shifted_arguments() -> None:
    found = mellin_transform_series(x**2 * exp(-3 * x), x, s)
    assert found is not None and found.strip == (-2, oo)
    assert _agree(found.transform, 3**(-s - 2) * gamma(s + 2), Rational(1, 2))
    # a negative scale is read as a positive one
    found = mellin_transform_series(exp(-2 * x**2), x, s)
    assert found is not None and _agree(found.transform, 2**(-s / 2) * gamma(s / 2) / 2, Rational(3, 2))
    found = mellin_transform_series(log(1 + 2 * x**2), x, s)
    assert found is not None and found.strip == (-2, 0)
    table = mellin_transform(log(1 + 2 * x**2), x, s)
    assert table is not None and _agree(found.transform, table.transform, -S.Half)


def test_method_of_brackets_against_marichev() -> None:
    # GR 3.944.5: Integral(x^(s-1) exp(-x) sin(x)) = Gamma(s) sin(pi s/4) / 2^(s/2)
    found = method_of_brackets(exp(-x) * sin(x), x, s)
    assert found is not None
    for point in (Rational(1, 3), Rational(3, 2)):
        assert _agree(found.transform, gamma(s) * sin(pi * s / 4) / 2**(s / 2), point)
        product = mellin_integrate(x**(point - 1) * exp(-x) * sin(x), x)
        assert product is not None and _agree(found.transform, product.value, point)
    assert found.strip == (-1, oo)
    # exp(-x^2) cos(2x): the bracket leaves a hypergeometric function
    found = method_of_brackets(exp(-x**2) * cos(2 * x), x, s)
    assert found is not None and found.strip == (0, oo)
    for point in (Rational(1, 3), Rational(3, 2)):
        product = mellin_integrate(x**(point - 1) * exp(-x**2) * cos(2 * x), x)
        assert product is not None and _agree(found.transform, product.value, point)
    # the public entry dispatches on the number of factors
    entry = mellin_transform_series(exp(-x) * sin(x), x, s)
    assert entry is not None and _agree(entry.transform, gamma(s) * sin(pi * s / 4) / 2**(s / 2), S.Half)


def test_method_of_brackets_gives_up() -> None:
    # three factors, a factor without a series, a non-closing sum
    assert mellin_transform_series(exp(-x) * sin(x) * cos(x), x, s) is None
    assert method_of_brackets(exp(-x) * besselj(0, x), x, s) is None
    assert method_of_brackets(exp(-x) * erf(x), x, s) is None


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(mellin_transform_series)([1], x, s))
