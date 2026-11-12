"""Tests of the native port of SymPy's ``solveset``
(sympy_extras.solvers.solveset): the expected sets are SymPy's (SymPy 1.14)
or hand computations, the elements checked by substitution."""
from __future__ import annotations

from sympy_extras.core import Eq, I, Ne, Rational, Symbol, oo, pi, sqrt
from sympy_extras.functions import cos, exp, log, sin, sinh, tan, zeta
from sympy_extras.sets import Complexes, ConditionSet, EmptySet, FiniteSet, ImageSet, Interval, Reals
from sympy_extras.sets.sets import Union, interval
from sympy_extras.solvers.solveset import real_polynomial_solutions, solveset

x, a, n = Symbol('x'), Symbol('a'), Symbol('n')


def test_real_polynomials_with_irrational_coefficients() -> None:
    found = solveset(Eq(x**2*(sqrt(2) - 1)**2 - 1, 0), x, Reals)
    assert isinstance(found, FiniteSet) and len(found.elements) == 2
    for r in found.elements:
        assert abs(complex((r**2*(sqrt(2) - 1)**2 - 1).evalf(30))) < 1e-25
    # the biquadratic of SymPy's {-sqrt(2*sqrt(2) + 3), sqrt(2*sqrt(2) + 3)}
    quartic = 8*x**4 + x**2*(-48 - 32*sqrt(2)) + 96*sqrt(2) + 136
    found = solveset(Eq(quartic, 0), x, Reals)
    assert isinstance(found, FiniteSet) and len(found.elements) == 2
    for r in found.elements:
        assert abs(complex(quartic.subs(x, r).evalf(30))) < 1e-20
    assert solveset(Eq(2*x - I, 0), x, Reals) == EmptySet
    assert real_polynomial_solutions(x**3 - pi, x, '==') == FiniteSet(pi**Rational(1, 3))


def test_real_inequalities() -> None:
    r = sqrt(sqrt(2) + 2)
    assert solveset(x**2 - 2 > sqrt(2), x, Reals) == Union(interval(-oo, -r, True, True), interval(r, oo, True, True))
    assert real_polynomial_solutions(x**2 - sqrt(2), x, '<') == interval(-2**Rational(1, 4), 2**Rational(1, 4), True, True)
    # a parametric even binomial: SymPy's Interval.open(-exp(a/2), exp(a/2))
    b = Symbol('b', real=True)
    found = solveset(exp(b) > x**2, x, Reals)
    assert isinstance(found, Interval) and found.left_open and found.right_open


def test_unsolved_relations_are_condition_sets() -> None:
    assert isinstance(solveset(Eq(x, cos(x)), x, Reals), ConditionSet)
    assert isinstance(solveset(x**2/4 - cos(x) >= 0, x, Reals), ConditionSet)
    assert isinstance(solveset(Ne(x*exp(x), 1), x, Reals), ConditionSet)
    assert isinstance(solveset(Eq(zeta(x), 0), x, Reals), ConditionSet)


def test_complex_equations() -> None:
    assert solveset(Eq(2*x - I, 0), x, Complexes) == FiniteSet(I/2)
    assert str(solveset(Eq(sinh(x), I), x, Complexes)) == 'ImageSet(Lambda(_n, 2*_n*I*pi + I*pi/2), Integers)'
    found = solveset(Eq(sin(a*x), 0), x, Complexes)
    assert isinstance(found, Union) and all(isinstance(s, ImageSet) for s in found.sets)
    # a radical equation without solution (SymPy: EmptySet)
    c = Symbol('c')
    assert solveset(Eq(sqrt(2)*sqrt(c) - sqrt(2*c*x**2 - 2*c*x)/(x - 1), 0), x, Complexes) == EmptySet
    # the roots of a numerical quartic are written a + b*I: Ferrari's
    # formulas put a square root on its branch cut, where evalf picked the
    # conjugate (the bug: a "root" with residual 1.87 after the conversion)
    found = solveset(Eq(x**4 + x + 1, 0), x, Complexes)
    assert isinstance(found, FiniteSet) and len(found.elements) == 4
    for r in found.elements:
        assert abs(complex((r**4 + r + 1).evalf(40))) < 1e-25


def test_periodic_and_logarithmic_real_equations() -> None:
    found = solveset(Eq(sin(pi/n), 0), n, Reals)
    assert str(found) == 'Union(ImageSet(Lambda(_n, 1/(2*_n)), Integers), ImageSet(Lambda(_n, 1/(2*_n + 1)), Integers))'
    # SymPy's solveset gave {-1, 2}: -1 is not a solution over the reals
    assert solveset(Eq(log(x) + log(x - 1), log(2)), x, Reals) == FiniteSet(2)
    assert isinstance(solveset(Eq(cos(x/2), 0), x, Reals), Union)
    assert solveset(Eq(tan(x), 1), x, Reals) != EmptySet
