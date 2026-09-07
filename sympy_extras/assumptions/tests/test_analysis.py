from __future__ import annotations

from sympy import (S, sin, cos, exp, log, sqrt, pi, E, Abs, Interval, Union, Rational,
    Eq, Ne, atan, tanh, Max, Piecewise)
from sympy.abc import x, y

from sympy_extras.assumptions import ask, refine, simplify, element, solve
from sympy_extras.assumptions.analysis import (certified_sign, domain_of, sign_on,
    decide_relational)
from sympy_extras.assumptions.facts import Facts
from sympy_extras.settings import settings, configure


def test_certified_sign() -> None:
    assert certified_sign(S(3)) == 1 and certified_sign(S(0)) == 0 and certified_sign(S(-2)) == -1
    assert certified_sign(E - 2) == 1
    assert certified_sign(sin(3) - Rational(1, 7)) == -1  # sin(3) = 0.1411... < 1/7
    assert certified_sign(log(2) - 1) == -1
    assert certified_sign(sqrt(2) - Rational(1414213562, 10**9)) == 1
    assert certified_sign(sqrt(2)**2 - 2) == 0
    assert certified_sign(x) is None
    assert certified_sign(S.ImaginaryUnit) is None
    assert certified_sign(S.Infinity) == 1 and certified_sign(S.NegativeInfinity) == -1
    with configure(numerical_checks=False):
        # SymPy's own knowledge (which includes its numerical evaluation of
        # constants) is still used; only the high precision evaluation is off
        assert certified_sign(S(3)) == 1
        assert certified_sign(E - 2) == 1
        assert certified_sign(sin(3) - Rational(1, 7)) == -1


def test_domain_of() -> None:
    assert domain_of(x, Facts([x > 0])) == Interval.open(0, S.Infinity)
    assert domain_of(x, Facts([(x > 0) & (x < pi)])) == Interval.open(0, pi)
    assert domain_of(x, Facts([x >= 1, x <= 2])) == Interval(1, 2)
    assert domain_of(x, Facts([x**2 < 4, x > 0])) == Interval.open(0, 2)
    assert domain_of(x, Facts([Ne(x, 0), x > -1, x < 1])) == Union(Interval.open(-1, 0), Interval.open(0, 1))
    assert domain_of(x, Facts([element(x, Interval(0, 1))])) == Interval(0, 1)
    assert domain_of(x, Facts([element(x, S.Reals)])) == S.Reals
    assert domain_of(x, Facts([x > y])) is None
    assert domain_of(x, Facts([y > 0])) is None
    assert domain_of(x, Facts([sin(x) > 0])) is None


def test_sign_on() -> None:
    assert sign_on(sin(x), x, Interval.open(0, pi)) == frozenset([1])
    assert sign_on(sin(x), x, Interval(0, pi)) == frozenset([0, 1])
    assert sign_on(sin(x), x, Interval.open(0, 4)) == frozenset([-1, 0, 1])
    assert sign_on(exp(x) - 2, x, Interval.open(1, S.Infinity)) == frozenset([1])
    assert sign_on(exp(x) - 2, x, S.Reals) == frozenset([-1, 0, 1])
    assert sign_on(log(x), x, Interval.open(0, 1)) == frozenset([-1])
    assert sign_on(cos(x) - 1, x, Interval.open(-1, 1)) == frozenset([-1, 0])
    assert sign_on(x*exp(x) - 1, x, Interval.open(1, S.Infinity)) == frozenset([1])  # by monotonicity
    assert sign_on(x - sin(x), x, Interval.open(0, S.Infinity)) == frozenset([1])
    assert sign_on(2 + sin(x), x, S.Reals) == frozenset([1])  # by bounds
    assert sign_on(1 - cos(x), x, S.Reals) == frozenset([0, 1])
    assert sign_on(1/x, x, Interval.open(-1, 1)) is None  # not continuous
    assert sign_on(S(5), x, Interval(0, 1)) == frozenset([1])


def test_decide_relational() -> None:
    facts = Facts([(x > 0) & (x < pi)])
    assert decide_relational(sin(x) > 0, facts) is True
    assert decide_relational(sin(x) < 0, facts) is False
    assert decide_relational(sin(x) >= 0, facts) is True
    assert decide_relational(Eq(sin(x), 0), facts) is False
    assert decide_relational(Ne(sin(x), 0), facts) is True
    assert decide_relational(sin(x) > Rational(1, 2), facts) is None
    assert decide_relational(sin(x) > y, Facts([x > 0, y > 0])) is None


def test_ask_beyond_polynomials() -> None:
    assert ask(sin(x) > 0, (x > 0) & (x < pi)) is True
    assert ask(sin(x) > 0, (x > 0) & (x < 4)) is None
    assert ask(sin(x) < 0, (x > pi) & (x < 2*pi)) is True
    assert ask(exp(x) > 2, x > 1) is True
    assert ask(exp(x) > 3, x > 1) is None
    assert ask(log(x) < 0, (x > 0) & (x < 1)) is True
    assert ask(x*exp(x) > 1, x > 1) is True
    assert ask(cos(x) < 1, Ne(x, 0) & (x > -1) & (x < 1)) is True
    assert ask(atan(x) > 0, x > 0) is True
    assert ask(tanh(x) < 1, domain=S.Reals) is True
    assert ask(x > sin(x), x > 0) is True
    assert ask(sqrt(x) < x, x > 1) is True
    assert ask(exp(x) >= x + 1, domain=S.Reals) is True


def test_refine_beyond_polynomials() -> None:
    assert refine(Abs(sin(x)), (x > 0) & (x < pi)) == sin(x)
    assert refine(Abs(sin(x)), (x > pi) & (x < 2*pi)) == -sin(x)
    assert refine(Abs(exp(x) - 2), x > 1) == exp(x) - 2
    assert refine(Abs(log(x)), (x > 0) & (x < 1)) == -log(x)
    assert refine(sqrt(sin(x)**2), (x > 0) & (x < pi)) == sin(x)
    assert refine(Max(x, sin(x)), x > 0) == x
    assert refine(Piecewise((1, exp(x) > 2), (2, True)), x > 1) == 1
    assert simplify(Abs(log(x))/log(x), x > 1) == 1
    assert refine(Abs(sin(x)), x > 0) == Abs(sin(x))


def test_solve_and_settings() -> None:
    assert solve(sin(x) - Rational(1, 2), x, (x > 0) & (x < 1)) == {pi/6}
    with configure(numerical_checks=False):
        assert settings.numerical_checks is False
        # exact knowledge still works
        assert ask(exp(x) > 2, x > 1) is True
        assert refine(Abs(sin(x)), (x > 0) & (x < pi)) == sin(x)
    assert settings.numerical_checks is True
    with configure(precision=50, timeout=5.0):
        assert settings.precision == 50 and settings.timeout == 5.0
    with configure(no_timeout=True):
        assert settings.timeout is None
    assert settings.timeout == 30.0
    assert repr(settings) == "Settings(numerical_checks=True, precision=30, timeout=30.0)"
