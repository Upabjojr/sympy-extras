"""Tests of the structure theorem: the tower, the canonical form, the zero test."""
from __future__ import annotations

import random

from sympy import (E, I, Integer, Rational, S, Symbol, acos, acosh, acot, acoth, asin, asinh, atan, atanh, cos, cosh,
                   exp, gamma, log, pi, sin, sinh, sqrt, symbols, tan, tanh)
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import as_expr
from sympy_extras.simplify import ElementaryTower, NotElementary, canonical_form, equal, is_zero

x, y = symbols('x y')


def _close(a: Expr, b: Expr, values: dict[Symbol, Expr]) -> bool:
    left = complex(as_expr(a.xreplace(values)).evalf(30))
    right = complex(as_expr(b.xreplace(values)).evalf(30))
    return abs(left - right) <= 1e-9 * (1 + abs(left))


def test_exponentials_are_a_homomorphism() -> None:
    assert is_zero(exp(x + y) - exp(x) * exp(y)) is True
    assert is_zero(exp(2 * x) - exp(x)**2) is True
    assert is_zero(exp(x) * exp(-x) - 1) is True
    assert is_zero(exp(x / 2)**2 - exp(x)) is True
    # a certified tower: a nonzero numerator is a proof
    assert is_zero(exp(x) - x - 1) is False
    assert is_zero(exp(x) - exp(2 * x)) is False
    tower = ElementaryTower()
    tower.element(exp(x) + exp(2 * x) + exp(x / 3))
    assert tower.certified
    kinds = sorted(g.kind for g in tower.generators)
    assert kinds.count('exp') == 1 and kinds.count('variable') == 1


def test_the_root_of_an_exponential_met_later() -> None:
    # exp(x) after exp(2*x): the root of index two of the generator, and
    # exp(x/3) after both: a root of index six which replaces it
    tower = ElementaryTower()
    a = tower.element(exp(2 * x))
    b = tower.element(exp(x))
    c = tower.element(exp(x / 3))
    assert tower.vanishes(b * b - a) and tower.vanishes(c**6 - a) and tower.vanishes(c**3 - b)
    assert tower.certified
    assert [g.kind for g in tower.generators] == ['variable', 'exp', 'root', 'root']
    assert tower.generators[2].eliminated is not None and tower.generators[3].index == 6


def test_logarithms_need_their_branch() -> None:
    assert is_zero(log(x**2) - 2 * log(x), x > 0) is True
    assert is_zero(log(x**2) - 2 * log(x)) is False            # x = -1
    assert is_zero(log(x * y) - log(x) - log(y), [x > 0, y > 0]) is True
    # one positive factor is enough: log(u*w) = log(u) + log(w) for u > 0
    assert is_zero(log(x * y) - log(x) - log(y), x > 0) is True
    assert is_zero(log(x * y) - log(x) - log(y)) is False
    assert is_zero(log(1 / x) + log(x), x > 0) is True
    assert is_zero(log(1 / x) + log(x)) is False               # x = -1: 2*I*pi
    assert is_zero(log(-x) - log(x) - I * pi, x > 0) is True
    assert is_zero(log(exp(x)) - x) is False                   # x = 4*I
    p = Symbol('p', positive=True)
    assert is_zero(log(exp(p)) - p) is True
    # a dependent logarithm whose branch is not known leaves the tower uncertified
    tower = ElementaryTower()
    u = tower.element(log(x**2) - 2 * log(x))
    assert not tower.vanishes(u) and not tower.certified


def test_radicals() -> None:
    assert is_zero(sqrt(x)**2 - x) is True
    assert is_zero((x**Rational(1, 4))**2 - sqrt(x)) is True
    assert is_zero(sqrt(x) * x**Rational(1, 3) - x**Rational(5, 6)) is True
    assert is_zero(1 / (sqrt(x) + 1) - (sqrt(x) - 1) / (x - 1)) is True
    assert is_zero(log(sqrt(x)) - log(x) / 2) is True
    assert is_zero(sqrt(x**2) - x) is False and is_zero(sqrt(x**2) - x, x > 0) is True
    assert is_zero(sqrt(x**2) + x, x < 0) is True
    assert is_zero(sqrt(x) * sqrt(y) - sqrt(x * y)) is False   # x = y = -1
    assert is_zero(sqrt(x) * sqrt(y) - sqrt(x * y), [x > 0, y > 0]) is True
    assert is_zero(sqrt(x + 1) * sqrt(x - 1) - sqrt(x**2 - 1), x > 1) is True
    assert is_zero(sqrt(x + 1) * sqrt(x - 1) - sqrt(x**2 - 1)) is False
    assert canonical_form((sqrt(x) + 1)**2 - x - 1) == 2 * sqrt(x)
    # Kummer theory: sqrt(x), sqrt(y) and sqrt(x + y) are independent,
    # sqrt(x*y) with no sign known is not
    tower = ElementaryTower()
    tower.element(sqrt(x) + sqrt(y) + sqrt(x + y) + x**Rational(1, 3))
    assert tower.certified
    tower = ElementaryTower()
    tower.element(sqrt(x) + sqrt(y) + sqrt(x * y))
    assert not tower.certified


def test_constants() -> None:
    assert is_zero(exp(I * pi) + 1) is True
    assert is_zero(E**(I * pi / 3) - Rational(1, 2) - sqrt(3) * I / 2) is True
    assert is_zero(log(6) - log(2) - log(3)) is True
    assert is_zero(log(8) / log(2) - 3) is True
    assert is_zero(log(Rational(-4, 9)) - 2 * log(2) + 2 * log(3) - I * pi) is True
    assert is_zero(sqrt(2) * sqrt(3) - sqrt(6)) is True
    assert is_zero(exp(log(2) / 2) - sqrt(2)) is True
    assert is_zero(log(I) - I * pi / 2) is True
    assert is_zero(log(1 + I) - log(2) / 2 - I * pi / 4) is True
    assert is_zero(log(3 + 2 * sqrt(2)) - 2 * log(1 + sqrt(2))) is True
    assert is_zero(sin(pi / 6) - S.Half) is True and is_zero(cos(pi / 5) - (1 + sqrt(5)) / 4) is True
    assert is_zero(2**x * 3**x - 6**x) is True and is_zero(4**x - 2**(2 * x)) is True
    # independent under Schanuel's conjecture
    assert is_zero(exp(pi) - pi**E) is False
    assert is_zero(pi - Rational(22, 7)) is False and is_zero(E - Rational(19, 7)) is False
    tower = ElementaryTower()
    tower.element(pi + E + log(2) + exp(sqrt(2)))
    assert tower.certified and len(tower.generators) == 4


def test_trigonometric_and_inverse_functions() -> None:
    assert is_zero(sin(x)**2 + cos(x)**2 - 1) is True
    assert is_zero(sin(x + y) - sin(x) * cos(y) - cos(x) * sin(y)) is True
    assert is_zero(cos(3 * x) - 4 * cos(x)**3 + 3 * cos(x)) is True
    assert is_zero(tanh(x / 2) - sinh(x) / (cosh(x) + 1)) is True
    assert is_zero(cosh(x)**2 - sinh(x)**2 - 1) is True
    assert is_zero(sin(x)**4 - cos(x)**4 - cos(2 * x)) is False
    assert is_zero(asin(x) + acos(x) - pi / 2) is True
    assert is_zero(asinh(x) - log(x + sqrt(x**2 + 1))) is True
    assert is_zero(atan(x) + atan(1 / x) - pi / 2, x > 0) is True
    assert is_zero(atan(x) + atan(1 / x) + pi / 2, x < 0) is True
    assert is_zero(atan(x) + atan(1 / x) - pi / 2) is False
    assert is_zero(atanh(x) - log((1 + x) / (1 - x)) / 2, [x > -1, x < 1]) is True


def test_a_pole_inside_the_region_splits_it() -> None:
    # the bug: atan(x) + atan(1/x) - pi/2 was proved for every real x from
    # its value at one positive point, though it is -pi for x < 0: the
    # locally constant difference is constant on a region only when
    # everything is analytic on it, and 1/x has its pole at 0
    real = Symbol('t', real=True)
    positive, negative = Symbol('p', positive=True), Symbol('n', negative=True)
    assert is_zero(atan(real) + atan(1 / real) - pi / 2) is False
    assert is_zero(atan(positive) + atan(1 / positive) - pi / 2) is True
    assert is_zero(atan(negative) + atan(1 / negative) + pi / 2) is True
    assert is_zero(atan(x) + atan(1 / x) - pi / 2, [x > -1, x < 1]) is False
    assert is_zero(sqrt(real**2) - real) is False and is_zero(sqrt(positive**2) - positive) is True


def test_the_rewritings_are_the_principal_values() -> None:
    # the module writes the inverse functions with logarithms: SymPy's
    # rewriting must agree with the principal values off the real axis too
    points = [Rational(3, 10), Rational(-7, 10), Rational(5, 2), Rational(-31, 10), Rational(2, 5) + 9 * I / 10,
              Rational(-6, 5) - I / 2, 3 * I / 2, -2 * I]
    for f in (asin, acos, atan, acot, asinh, acosh, atanh, acoth):
        for point in points:
            assert _close(f(x), as_expr(f(x).rewrite(log)), {x: point}), (f, point)
    for g in (sin, cos, tan, sinh, cosh, tanh):
        for point in points:
            assert _close(g(x), as_expr(g(x).rewrite(exp)), {x: point}), (g, point)


def test_the_canonical_form_has_the_value_of_the_expression() -> None:
    # random expressions: the canonical form agrees with the expression at
    # complex points without assumptions, at positive points under them
    rng = random.Random(7)
    atoms: list[Expr] = [x, y, x, Integer(2), Rational(1, 2), Integer(-1), I, pi]

    def build(depth: int) -> Expr:
        if depth == 0 or rng.random() < 0.3:
            return rng.choice(atoms)
        kind = rng.randrange(9)
        u = build(depth - 1)
        if kind == 0:
            return exp(u)
        if kind == 1:
            return log(u)
        if kind == 2:
            return sqrt(u)
        if kind == 3:
            return sin(u)
        if kind == 4:
            return atan(u)
        if kind == 5:
            return as_expr(u**Rational(1, 3))
        v = build(depth - 1)
        return as_expr(u + v) if kind == 6 else as_expr(u * v) if kind == 7 else as_expr(u / v)

    checked = 0
    for _ in range(25):
        e = build(3)
        for assumptions, positive in ((None, False), ([x > 0, y > 0], True)):
            try:
                c = canonical_form(e, assumptions)
            except (NotElementary, ZeroDivisionError):
                continue
            for _ in range(2):
                values: dict[Symbol, Expr] = {
                    s: as_expr(Rational(rng.randint(1, 400), 100) if positive
                               else Rational(rng.randint(-300, 300), 100) + I * Rational(rng.randint(-400, 400), 100))
                    for s in (x, y)}
                try:
                    left = complex(as_expr(e.xreplace(values)).evalf(30))
                    right = complex(as_expr(c.xreplace(values)).evalf(30))
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                if left != left or right != right or abs(left) > 1e40:
                    continue
                checked += 1
                assert abs(left - right) <= 1e-8 * (1 + abs(left)), (e, c, values)
    assert checked > 40


def test_outside_the_elementary_functions() -> None:
    raises(NotElementary, lambda: canonical_form(gamma(x)))
    raises(NotElementary, lambda: canonical_form(x + 0.5))
    raises(NotElementary, lambda: canonical_form(log(x - x)))
    assert is_zero(gamma(x) - gamma(x)) is True                 # SymPy cancels it before the tower sees it
    assert is_zero(gamma(x + 1) - x * gamma(x)) is None
    assert is_zero(gamma(x) - 1) is False                       # a sample point tells
    assert equal(tan(x), sin(x) / cos(x)) is True and equal(sin(x), x) is False
    raises(TypeError, lambda: untyped(canonical_form)([1]))
    tower, other = ElementaryTower(), ElementaryTower()
    raises(ValueError, lambda: tower.element(x) + other.element(x))
