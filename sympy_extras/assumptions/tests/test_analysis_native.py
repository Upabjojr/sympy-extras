"""The native analysis and interval arithmetic on expressions of the core,
checked against hand computations (the SymPy faces are tested in
test_analysis.py and test_intervals.py)."""
from __future__ import annotations

from sympy_extras.core import E, Integer, Rational, Symbol, oo, pi, sqrt
from sympy_extras.core.relational import Gt
from sympy_extras.functions import cos, exp, log, sin
from sympy_extras.sets import Interval, Reals
from sympy_extras.assumptions.facts import Facts
from sympy_extras.assumptions.analysis import (certified_compare, certified_sign, decide_relational, domain_of,
    sign_on, signs_multivariate)
from sympy_extras.assumptions.intervals import (Box, corner_value, evaluate, interval_sign, isolate_signs, range_on_box,
    signs_on_box)
from sympy_extras.settings import configure

x, y = Symbol('x'), Symbol('y')


def test_native_certified_sign() -> None:
    assert certified_sign(E - 2) == 1 and certified_sign(log(2) - 1) == -1
    assert certified_sign(sin(3) - Rational(1, 7)) == -1   # sin(3) = 0.14112...
    assert certified_sign(pi - Rational(355, 113)) == -1   # 3.14159265358979 < 3.14159292035398
    assert certified_sign(x) is None and certified_sign(oo) == 1 and certified_sign(-oo) == -1
    # a hidden zero of an algebraic number, from its minimal polynomial
    # (SymPy's is_zero knows it): no sign is read from the noise
    hidden = (1 + sqrt(2))**2 - 2*sqrt(2) - 3
    assert certified_sign(hidden) == 0
    assert certified_sign(sin(1)**2 + cos(1)**2 - 1) is None
    with configure(numerical_checks=False):
        assert certified_sign(hidden) == 0
    assert certified_compare(sqrt(2), Rational(3, 2)) == -1 and certified_compare(oo, oo) == 0


def test_native_intervals() -> None:
    box: Box = {x: (Integer(0), Integer(1)), y: (Integer(0), Integer(2))}
    enclosure = evaluate(exp(x) - 1, {x: (Integer(0), Integer(1))})
    assert enclosure is not None and enclosure.a <= 0 and enclosure.b >= 1.718281828
    assert interval_sign(evaluate(-exp(x) - 1, {x: (-oo, oo)})) == -1
    assert evaluate(log(x), {x: (Integer(-1), Integer(1))}) is None
    assert signs_on_box(x - y, box) is None
    assert range_on_box(x*y + 1, box) == Interval(1, 3)
    # the limit at the infinite corner is the core's
    assert corner_value(x*exp(-x) + y, {x: oo, y: Integer(2)}) == Integer(2)
    assert isolate_signs(exp(x) - 2, x, Interval(log(2), 3)) == frozenset([0, 1])
    assert isolate_signs(exp(x) - 2, x, Interval(log(2), 3, True, False)) == frozenset([1])


def test_native_signs_of_rational_functions() -> None:
    # zeros and poles by solveset_rational: sqrt(2) is the only zero in (0, 10)
    assert sign_on(x**2 - 2, x, Interval(0, 10, True, True)) == frozenset([-1, 0, 1])
    assert sign_on(x**2 + 1, x, Reals) == frozenset([1])
    assert sign_on(1/(x**2 - 2), x, Interval(0, 1, True, True)) == frozenset([-1])
    assert sign_on(1/(x**2 - 2), x, Interval(0, 2, True, True)) is None   # a pole at sqrt(2)
    assert sign_on(1/x - 1/x**2, x, Interval(0, 3, True, True)) == frozenset([-1, 0, 1])
    assert decide_relational(Gt(x**2, 1), Facts([x > 2])) is True
    assert decide_relational(Gt(x, 5), Facts([x > 2])) is None


def test_native_analysis_through_sympy_steps() -> None:
    from sympy_extras.interop import from_sympy  # solveset, continuous_domain and the CAD are SymPy's
    assert from_sympy(1) == Integer(1)
    assert sign_on(sin(x), x, Interval(0, pi, True, True)) == frozenset([1])
    assert sign_on(x*exp(x) - 1, x, Interval(1, oo, True, True)) == frozenset([1])
    assert domain_of(x, Facts([x**2 < 4, x > 0])) == Interval(0, 2, True, True)
    # the end sqrt(2) is CRootOf(x**2 - 2, 1) from the CAD: SymPy intersects
    domain = domain_of(x, Facts([x**2 < 2, x > 0]))
    assert isinstance(domain, Interval) and domain.left_open and domain.right_open and domain.start == 0
    end = domain.end
    assert certified_compare(end, Rational(1414, 1000)) == 1 and certified_compare(end, Rational(1415, 1000)) == -1
    assert decide_relational(Gt(sin(x), 0), Facts([x > 0, x < pi])) is True
    assert signs_multivariate(exp(x*y) - 1, [x, y], Facts([x > 0, x < 1, y >= 2])) == frozenset([1])


def test_roots_of_sympy_have_values_in_sums() -> None:
    import sympy
    from sympy_extras.interop import expr_from_sympy
    from sympy_extras.core import Float
    root = expr_from_sympy(sympy.CRootOf(sympy.Symbol('x')**5 - sympy.Symbol('x') + 1, 0))
    # CRootOf (not ported, a SymPy wrapper) had no value inside a sum of the
    # core: (r + 2).evalf() stayed CRootOf(...) + 2.0, so the zero of
    # x**5 - x + 1 found by solveset could not be ordered and the signs
    # were not found (None instead of {-1, 0, 1})
    assert isinstance((root + 2).evalf(30), Float)
    assert certified_sign(root + 2) == 1 and certified_sign(root + 1) == -1   # r = -1.1673...
    assert sign_on(x**5 - x + 1, x, Reals) == frozenset([-1, 0, 1])


def test_zeros_of_a_variable_with_assumptions() -> None:
    # the zeros are those of a real variable on the domain given: with a
    # positive symbol, the native transcendental solver dropped the zero
    # a = 0 of cosh(2*a) - 1 and sign_on on the reals gave {0} (the sign at
    # the one sample point), not {0, 1}
    from sympy_extras.functions import cosh
    from sympy_extras.assumptions.analysis import _finite_zeros
    a = Symbol('a', positive=True)
    assert sign_on(cosh(2*a) - 1, a, Reals) == frozenset([0, 1])
    assert _finite_zeros(cosh(2*a) - 1, a, Reals) == [Integer(0)]


def test_continuity_and_zeros_of_elementary_functions() -> None:
    # the continuity and the zeros of the functions which are not rational
    # (SymPy's continuous_domain and solveset before), each against a hand
    # computation
    from sympy_extras.functions import Abs, acos, asin, atan, tan
    from sympy_extras.assumptions.analysis import _continuous_on, _finite_zeros, _has_discrete_zeros
    x = Symbol('x')
    assert _continuous_on(log(x) + sqrt(x - 1), x, Interval(1, 2))
    assert not _continuous_on(log(x), x, Interval(0, 1))
    assert not _continuous_on(sqrt(x - 1), x, Interval(0, 2))
    assert _continuous_on(tan(x), x, Interval(-1, 1)) and not _continuous_on(tan(x), x, Interval(1, 2))
    assert _continuous_on(asin(x/2) + acos(x/3), x, Interval(-2, 2))
    assert not _continuous_on(asin(x), x, Interval(0, 2))
    assert _continuous_on(Abs(x) + atan(x) + 2**x, x, Reals)
    # 1 + sin(x)/x is positive for x >= 1: log of it is continuous there
    assert _continuous_on(log(1 + sin(x)/x), x, Interval(1, oo))
    assert _finite_zeros(exp(x)*sin(x) + exp(x)*cos(x), x, Interval(0, 5, True, True)) == [3*pi/4]
    zeros = _finite_zeros(sin(x), x, Interval(0, 2*pi))
    assert zeros is not None and sorted(zeros, key=float) == [0, pi, 2*pi]
    assert _finite_zeros(sin(x), x, Reals) is None
    assert _has_discrete_zeros(sin(x) - Rational(1, 2), x, Reals)


def test_zero_of_a_harmonic_combination_at_an_end() -> None:
    # sin(x) - tan(1)*cos(x) vanishes at 1: the half-angle substitution
    # wrote the zero as 2*atan((sqrt(1 + tan(1)**2) - 1)/tan(1)), which no
    # comparison told apart from 1, so the zeros on (0, 1) were None (and
    # antiderivative_integral(sin(x)**2*cot(x - 1), x, 0, 1) gave None)
    from sympy_extras.functions import tan
    from sympy_extras.assumptions.analysis import _finite_zeros
    x = Symbol('x')
    f = sin(x) - tan(Integer(1))*cos(x)
    assert _finite_zeros(f, x, Interval(0, 1, True, True)) == []
    assert _finite_zeros(f, x, Interval(0, 1)) == [Integer(1)]
    assert sign_on(f, x, Interval(0, 1, True, True)) == frozenset([-1])
