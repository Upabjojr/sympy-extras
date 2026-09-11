"""Tests of Frullani's theorem and Glasser's master theorem (Boros–Moll,
Irresistible Integrals, 5.2 and 13; Glasser 1983)."""
from __future__ import annotations

from sympy import symbols, exp, cos, atan, log, sqrt, pi, oo, S, simplify

from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.integrals import definite_integral
from sympy_extras.integrals.definite import verify_numerically
from sympy_extras.integrals.transformations import frullani, glasser, transformation_integral

x = symbols('x')
a, b = symbols('a b', positive=True)


def _same(u: ExprLike, v: ExprLike) -> bool:
    return simplify(as_expr(u) - as_expr(v)) == 0


def test_frullani() -> None:
    # Boros-Moll 5.2: (f(0) - f(oo)) log(b/a)
    found = frullani((exp(-a * x) - exp(-b * x)) / x, x)
    assert found is not None and _same(found.value, log(b / a))
    found = frullani((atan(a * x) - atan(b * x)) / x, x)
    assert found is not None and _same(found.value, -pi * log(b / a) / 2)
    # the bug: (exp(-a x) - exp(-b x))/x is one product in SymPy, not a sum
    # of two terms, and the scales -a, -b are negative: the shape was missed
    assert frullani((exp(-a * x) - exp(-b * x)) / x, x) is not None
    # no finite limit at infinity: not applicable (the difference kernels do it)
    assert frullani((cos(a * x) - cos(b * x)) / x, x) is None
    # not a Frullani shape
    assert frullani((exp(-a * x) - exp(-b * x**2)) / x, x) is None
    assert frullani(exp(-a * x) / x, x) is None


def test_glasser() -> None:
    # the Cauchy-Schlomilch transformation over the half line (even F)
    found = glasser(exp(-(x - 1 / x)**2), x, 0, oo)
    assert found is not None and found.value == sqrt(pi) / 2
    found = glasser(1 / (1 + (x - 1 / x)**2), x, 0, oo)
    assert found is not None and found.value == pi / 2
    # Glasser's theorem over the real line with two poles
    found = glasser(exp(-(x - 1 / (x - 1) - 2 / (x + 1))**2), x, -oo, oo)
    assert found is not None and found.value == sqrt(pi)
    assert verify_numerically(sqrt(pi), exp(-(x - 1 / (x - 1) - 2 / (x + 1))**2), x, -oo, oo) is not False
    # an odd F over the half line, or a negative coefficient, is not covered
    assert glasser(exp(-(x - 1 / x)**2) * (x - 1 / x), x, 0, oo) is None
    assert glasser(exp(-(x + 1 / x)**2), x, -oo, oo) is None


def test_through_the_driver() -> None:
    assert _same(definite_integral((exp(-a * x) - exp(-b * x)) / x, (x, 0, oo)), log(b / a))
    assert definite_integral(exp(-(x - 1 / x)**2), (x, 0, oo)) == sqrt(pi) / 2
    # the bug: the singularity of the map at x = 1 made the driver cut the
    # range first and the residue route gave a page of radicals for pi
    assert definite_integral(1 / (1 + (x - 1 / (x - 1))**2), (x, -oo, oo)) == pi
    assert transformation_integral(exp(-x), x, S.Zero, S.One) is None
