"""Tests of the SymPy face of the structure theorem
(:mod:`sympy_extras.simplify.structure_sympy`): the conversions around the
native tower, the steps which stay on SymPy, the agreement of the face
with the native functions, and the callers written on SymPy (the exact
tests of the integrals)."""
from __future__ import annotations

import random

from sympy import (E, I, Rational, S, Symbol, asin, atan, cos, cosh, erf, exp, gamma, log, pi, sin, sinh, sqrt, symbols,
                   tan, tanh)
from sympy.core.expr import Expr
from sympy.logic.boolalg import Boolean
from sympy.testing.pytest import raises

from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import core_boolean
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol, sympy_expr
from sympy_extras.simplify import structure as native
from sympy_extras.simplify.structure_sympy import (EXPONENTIAL, LOGARITHM, ElementaryTower, Generator, NotElementary,
                                                   _witness, canonical_form, is_zero)

x, y = symbols('x y')


def test_the_elements_of_the_face() -> None:
    tower = ElementaryTower()
    a, b = tower.element(x), tower.element(exp(x))
    assert tower.to_expr(a * b + a.times(Rational(3, 7))) == x * exp(x) + 3 * x / 7
    assert tower.to_expr((a + b) / (a - b)) == (x + exp(x)) / (x - exp(x))
    assert tower.to_expr(-a**2) == -x**2 and tower.to_expr(a**-1) == 1 / x
    assert tower.to_expr(tower.constant(Rational(-5, 3))) == Rational(-5, 3)
    assert tower.vanishes(tower.zero) and tower.to_expr(tower.one) == 1
    assert repr(a) == 'Element(x)' and isinstance(tower.to_expr(a), Expr)
    assert tower.as_number(tower.element(sqrt(2) * I)) == sqrt(2) * I and tower.as_number(a) is None
    raises(ValueError, lambda: a + ElementaryTower().element(x))
    raises(ValueError, lambda: tower.to_expr(ElementaryTower().element(x)))
    raises(TypeError, lambda: tower.element([1]))


def test_the_generators_of_the_face() -> None:
    tower = ElementaryTower()
    u = tower.element(exp(x) + exp(x / 2) + log(x))
    generators = tower.generators
    assert all(isinstance(g, Generator) for g in generators)
    assert [(g.kind, g.expression) for g in generators] == [('variable', x), ('exp', exp(x)), ('root', exp(x / 2)),
                                                            ('log', log(x))]
    root = generators[2]
    assert root.index == 2 and root.source == 0 and root.eliminated is None
    assert root.radicand is not None and tower.to_expr(root.radicand) == exp(x)
    assert root.scale is not None and tower.to_expr(root.scale) == 1
    assert generators[3].argument is not None and tower.to_expr(generators[3].argument) == x
    assert repr(generators[1]) == 'Generator(exp, exp(x))' and tower.certifies(u) and tower.certified
    v = tower.variable(x)
    assert tower.vanishes(tower.derivative(tower.exponential(v), 0) - tower.exponential(v))
    assert tower.vanishes(tower.logarithm(tower.exponential(v)) - v) is False      # x complex: no proof
    assert tower.is_constant(tower.element(sin(x)**2 + cos(x)**2))
    assert tower.to_expr(tower.normal(tower.element(sqrt(x)) ** 2)) == x


def test_the_face_agrees_with_the_native_functions() -> None:
    cases: list[tuple[Expr, list[Boolean]]] = [
        (exp(x + y) - exp(x) * exp(y) + sin(x)**2, []), (log(x * y) - log(x), [x > 0]),
        ((sqrt(x) + 1)**3, []), (tanh(x) + 1 / x, []), (atan(x) + atan(1 / x), [x > 0]),
        (cosh(x)**2 - sinh(x)**2, []), (pi + E + log(6), []), (x**y + 2**x, [])]
    for expression, facts in cases:
        assumptions = [core_boolean(f) for f in facts] or None
        native_form = native.canonical_form(core_expr(expression), assumptions)
        assert canonical_form(expression, facts or None) == sympy_expr(native_form), expression
        assert is_zero(expression - canonical_form(expression, facts or None), facts or None) is True
        assert native.is_zero(core_expr(expression), assumptions) == is_zero(expression, facts or None)


def test_the_numerical_steps_of_the_native_tower() -> None:
    # the sample points and the reliable values the native tower uses
    # (through the face until they were ported)
    from sympy_extras._numeric import reliable_complex
    from sympy_extras.integrals.conditions import sample_values
    a, b = core_symbol(x), core_symbol(y)
    point = sample_values([a, b], [core_boolean(x > 1), core_boolean(y < -2)], random.Random(3))
    assert point is not None and set(point) == {a, b}
    assert sympy_expr(point[a]) > 1 and sympy_expr(point[b]) < -2
    assert sample_values([a], [core_boolean(x > 1), core_boolean(x < 0)], random.Random(3)) is None
    value = reliable_complex(core_expr(exp(x) + 1), 30, {a: core_expr(Rational(1, 2))})
    assert value is not None and abs(value - complex(exp(Rational(1, 2)).evalf(30) + 1)) < 1e-12
    assert _witness(as_expr(log(x**2) - 2 * log(x)), None) and not _witness(as_expr(log(x**2) - 2 * log(x)), x > 0)


def test_what_the_face_refuses() -> None:
    raises(NotElementary, lambda: canonical_form(gamma(x)))
    raises(NotElementary, lambda: canonical_form(erf(x) + 1))
    assert is_zero(gamma(x + 1) - x * gamma(x)) is None
    assert is_zero(S.Zero) is True and is_zero(tan(x) - sin(x) / cos(x)) is True
    tower = ElementaryTower([x > 0, y > 0])
    tower.element(log(x * y))
    assert [g.kind for g in tower.generators].count(LOGARITHM) == 2
    assert all(g.kind != EXPONENTIAL for g in tower.generators)


def test_the_exact_test_of_is_antiderivative() -> None:
    from sympy_extras.integrals.indefinite import _exactly_zero, is_antiderivative
    assert _exactly_zero(as_expr(sin(x)**2 + cos(x)**2 - 1), None) is True
    assert _exactly_zero(as_expr(exp(x) - 1), None) is False
    assert _exactly_zero(as_expr(gamma(x) - gamma(x) + erf(x)), None) is False     # not elementary: no proof
    assert _exactly_zero(as_expr(log(x**2) - 2 * log(x)), [x > 0]) is True
    for F, f in ((x * atan(x) - log(x**2 + 1) / 2, atan(x)), (exp(x) * (sin(x) - cos(x)) / 2, exp(x) * sin(x)),
                 (2 * sqrt(x + 1) * (x - 2) / 3, x / sqrt(x + 1)), (log(tan(x / 2)), 1 / sin(x)),
                 (x * asin(x) + sqrt(1 - x**2), asin(x)), (tanh(x), 1 / cosh(x)**2)):
        assert _exactly_zero(as_expr(F.diff(x) - f), None) is True, F
        assert is_antiderivative(F, f, x) is True
    assert is_antiderivative(sin(x), sin(x), x) is False


def test_the_second_opinion_of_numerically_equal() -> None:
    # agreement at three sample points is evidence; two expressions which
    # the structure theorem proves different are different
    from sympy_extras.integrals.conditions_sympy import _provably_different, numerically_equal
    tiny = as_expr(exp(-10**6 * x**2 - 50))
    assert numerically_equal(tiny, S.Zero) is False                  # 1e-22 and less at every sample
    assert numerically_equal(exp(x), as_expr(exp(x) + Rational(1, 10**30))) is False
    assert numerically_equal(as_expr(sin(x)**2 + cos(x)**2), S.One) is True
    assert numerically_equal(as_expr(log(x**2)), as_expr(2 * log(x)), [x > 0]) is True
    assert numerically_equal(x, y) is False
    assert _provably_different(exp(x), as_expr(exp(x) + 1), None)
    assert not _provably_different(exp(x), exp(x), None)
    assert _provably_different(gamma(x), as_expr(gamma(x) + 1), None)          # SymPy leaves the difference 1
    assert not _provably_different(gamma(x), as_expr(2 * gamma(x) + 1), None)  # not elementary: no opinion
    assert not _provably_different(as_expr(log(x**2)), as_expr(2 * log(x)), None)  # uncertified: no opinion


def test_nothing_is_independent_after_a_dependent_generator() -> None:
    # the bug: with log(x) and the dependent log(x**2) in the tower,
    # exp(v*log(x**2)/2) was certified independent of exp(v*log(x)), and the
    # antiderivative -uppergamma(v/2, -a*x**2)/(2*(-a)**(v/2)) of
    # x**(v - 1)*exp(a*x**2) was "proved" wrong, its derivative being
    # written with (-a*x**2)**(v/2 - 1)
    from sympy_extras.integrals.conditions_sympy import _provably_different, numerically_equal
    v = Symbol('v', positive=True)
    a = Symbol('a', negative=True)
    derivative = as_expr(-a * x * (-a * x**2)**(v / 2 - 1) * exp(a * x**2) / (-a)**(v / 2))
    integrand = as_expr(x**(v - 1) * exp(a * x**2))
    tower = ElementaryTower()
    u = tower.element(derivative - integrand)
    assert not tower.vanishes(u) and not tower.certifies(u)
    assert not _provably_different(derivative, integrand, None)
    assert numerically_equal(derivative, integrand) is True
    assert is_zero(derivative - integrand, x > 0) is True
    # every exponential and logarithm after the dependent one is uncertified
    tower = ElementaryTower()
    tower.element(log(x) + log(x**2))
    before = len(tower.generators)
    tower.element(exp(y) + log(y + 1))
    assert all(not g.certified for g in tower.generators[before:] if g.kind in (EXPONENTIAL, LOGARITHM))
    assert tower.certifies(tower.element(log(x) + x))        # what came before stays certified
