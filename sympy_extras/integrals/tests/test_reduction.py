"""Tests of the reduction-based creative telescoping."""
from sympy import Expr, I, Rational, cancel, cos, exp, oo, pi, sqrt, symbols

from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.reduction import (HermiteReducer, hermite_reduce, reduction_integral, reduction_ode,
                                              reduction_telescoper)

x = symbols('x')
t = symbols('t', positive=True)


def _reduces(p: Expr, q: Expr) -> Expr:
    """``p*S == D(g) + r`` for the reducer of ``q``, returning ``r``."""
    reducer = HermiteReducer(q, x, t)
    g, r = reducer.reduce(p)
    assert cancel(p * reducer.shell - reducer.derivation(g) - r) == 0
    return r


def test_hermite_reduce_identity() -> None:
    for p, q in [(1 / x, 2 / x + 1 / (x - t)**2), (x**3 / (x**2 + t)**2, -2 * x),
                 (1 / ((x**2 + 1)**2 * (x - t)), 3 / (x - 1) + t / x**2),
                 (x**5 + 1 / (x - 2)**3, -2 * t / x**3 + 1 / x), (x**4, (1 - 3 * x**2) / (x**3 - x)),
                 (1 / (x**2 + 1), 2 / (x - I) + 3 / (x + I)), (x / (x**2 + t), 0)]:
        _reduces(p, q)
    for p, q in [(1 / x**2, -2 * x), (x**7 + 3 * x**2 / (x - t)**4, -2 * x)]:
        g, r = hermite_reduce(p, q, x, t)
        assert cancel(p - g.diff(x) - q * g - r) == 0


def test_hermite_reduce_integrable() -> None:
    # x exp(-x**2), x**3 exp(-x**2), (2 x t + 1) exp(x**2 t + x) are derivatives
    assert hermite_reduce(x, -2 * x, x)[1] == 0
    assert hermite_reduce(x**3, -2 * x, x)[1] == 0
    assert hermite_reduce(2 * x * t + 1, 2 * x * t + 1, x, t)[1] == 0
    # 1/(x**2 + 1)**2 is not the derivative of a rational function
    assert hermite_reduce(1 / (x**2 + 1)**2, 0, x, t)[1] != 0
    # but 2x/(x**2 + 1)**2 is
    assert hermite_reduce(2 * x / (x**2 + 1)**2, 0, x, t)[1] == 0
    # and exp(-x**2) has no elementary antiderivative
    assert hermite_reduce(1, -2 * x, x)[1] != 0


def test_shell() -> None:
    reducer = HermiteReducer(2 / x + 1 / (x - t)**2, x, t)
    assert reducer.shell == x**2
    assert reducer.B.as_expr().expand() == ((x - t)**2).expand()
    assert HermiteReducer(-3 / (x - 1) + t / x**2, x, t).shell == (x - 1)**-3
    assert HermiteReducer(2 / (x - I) + 3 / (x + I), x, t).shell == (x - I)**2 * (x + I)**3
    assert HermiteReducer(t / x, x, t).shell == 1


def test_reduction_telescoper() -> None:
    found = reduction_telescoper(1 / (x**2 + t**2), x, t)
    assert found is not None
    assert found.coefficients == [1, t] and found.certificate == -x and found.check()
    found = reduction_telescoper(exp(-x**2 - t**2 / x**2), x, t)
    assert found is not None
    assert found.coefficients == [-4, 0, 1] and found.check()
    found = reduction_telescoper(1 / (x**2 + t * x + 1), x, t)
    assert found is not None
    assert found.coefficients == [t, t**2 - 4] and found.check()
    found = reduction_telescoper(x**2 * exp(-x**2) * cos(2 * t * x), x, t)
    assert found is not None
    assert found.coefficients == [4 * t**3 - 6 * t, 2 * t**2 - 1] and found.check()
    found = reduction_telescoper(exp(-t * x) / (x**2 + 1), x, t)
    assert found is not None
    assert found.coefficients == [1, 0, 1] and found.check()
    assert reduction_telescoper(x**t * exp(-x), x, t) is None
    assert reduction_telescoper(exp(-x**2) * exp(2 * x * t), x, t, max_order=0) is None


def test_reduction_ode() -> None:
    equation = reduction_ode(exp(-x**2) * exp(2 * x * t), x, -oo, oo, t)
    assert equation is not None and equation.rhs == 0
    assert reduction_ode(x**t, x, 0, 1, t) is None


def test_reduction_integral() -> None:
    assert reduction_integral(exp(-x**2) * cos(2 * t * x), x, 0, oo, t) == ConditionalValue(sqrt(pi) * exp(-t**2) / 2)
    assert reduction_integral(1 / (x**2 + t**2)**2, x, 0, oo, t) == ConditionalValue(pi / (4 * t**3))
    assert reduction_integral(x**2 * exp(-t * x), x, 0, oo, t) == ConditionalValue(2 / t**3)
    assert reduction_integral(x**t * exp(-x), x, 0, oo, t) is None


def test_polynomial_part_exceptional_degree() -> None:
    # F = (x**2 + t)**(-5/2): the residue of q at infinity is -5, so E(x**3)
    # has a lower degree than E(x**2) + 1
    q = -5 * x / (x**2 + t)
    reducer = HermiteReducer(q, x, t)
    assert reducer.shell == 1
    assert reducer._image(3).degree() < reducer._image(2).degree() + 1
    for p in [x**2, x**3, x**4, x**5 + 1, Rational(1, 2) * x**2 + x]:
        _reduces(p, q)
    # the reduced forms are canonical: adding the image E(x**3) changes nothing
    image = reducer._image(3).as_expr()
    assert cancel(reducer.reduce(x**4)[1] - reducer.reduce(x**4 + image)[1]) == 0
    # integer residues at simple poles go to the shell instead
    assert hermite_reduce(x**2, -3 / x, x, t) == (0, x**2)
    assert hermite_reduce(x**3, -3 / x, x, t) == (x**4, 0)      # x**3 * x**-3 = (x * x**-3 * x**3)'
    assert hermite_reduce(x, -1 / x, x, t) == (x**2, 0)


def test_sums_of_terms() -> None:
    # similar terms are merged: a single order-two operator, not two
    # different order-one ones
    found = reduction_telescoper((x**4 + 1) * exp(-t * x**2) / (x**2 + 1), x, t)
    assert found is not None and found.check() and found.order == 2
    # non-similar terms: the common telescoper (each term alone has order one)
    found = reduction_telescoper(exp(-t * x**2) + exp(-t**2 * x**2), x, t)
    assert found is not None and found.check() and found.order == 2
    # a term that is integrable by itself does not raise the order
    found = reduction_telescoper(exp(-t * x) + exp(-t * x**2), x, t)
    assert found is not None and found.check() and found.order == 1
    found = reduction_telescoper(cos(t * x) * exp(-x), x, t)
    assert found is not None and found.check()


def test_reduced_forms_linear() -> None:
    reducer = HermiteReducer(-2 * x + t, x, t)
    g1, r1 = reducer.reduce(x**2 / (x - 1)**2)
    g2, r2 = reducer.reduce(3 * x**2 / (x - 1)**2 + x**4)
    g3, r3 = reducer.reduce(x**4)
    assert cancel(r2 - 3 * r1 - r3) == 0
