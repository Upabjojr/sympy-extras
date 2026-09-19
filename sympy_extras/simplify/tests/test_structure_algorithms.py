"""Tests of the algorithms of the structure theorem, one by one: the
exponential and the logarithm of an element, the logarithms of numbers,
the roots and their certification, the reading of expressions, the
decision and its witness."""
from __future__ import annotations

from sympy import (E, Abs, Float, I, Integer, Rational, S, Symbol, acos, acosh, acot, asin, asinh, atan, atanh, cos,
                   cosh, cot, coth, csc, erf, exp, gamma, log, pi, sec, sin, sinh, sqrt, symbols, tan, tanh)
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import as_expr
from sympy_extras.settings import configure
from sympy_extras.simplify import ElementaryTower, NotElementary, canonical_form, equal, is_zero
from sympy_extras.simplify.structure import EXPONENTIAL, LOGARITHM, ROOT, VARIABLE, _witness

x, y, z = symbols('x y z')
p, q = symbols('p q', positive=True)


def _kinds(tower: ElementaryTower) -> list[str]:
    return [g.kind for g in tower.generators]


def _value(expression: Expr, point: dict[Symbol, Expr]) -> complex:
    return complex(as_expr(expression.xreplace(point)).evalf(30))


# ---------------------------------------------------------------------------
# exponentials

def test_an_independent_exponential_is_a_generator() -> None:
    tower = ElementaryTower()
    theta = tower.exponential(tower.element(x))
    assert _kinds(tower) == [VARIABLE, EXPONENTIAL] and tower.to_expr(theta) == exp(x)
    # exp(x**2), exp(1/x), exp(sqrt(2)*x) are independent of exp(x) and of each other
    for argument in (x**2, 1 / x, sqrt(2) * x, x * y):
        before = _kinds(tower).count(EXPONENTIAL)
        tower.exponential(tower.element(argument))
        assert _kinds(tower).count(EXPONENTIAL) == before + 1, argument
    assert tower.certified
    assert tower.vanishes(tower.exponential(tower.zero) - tower.one)


def test_a_dependent_exponential_is_a_product_of_powers() -> None:
    tower = ElementaryTower()
    a, b = tower.exponential(tower.element(x)), tower.exponential(tower.element(y))
    count = len(tower.generators)
    for argument, expected in ((x + y, a * b), (2 * x - 3 * y, a**2 * b**-3), (-x, a**-1), (5 * x, a**5)):
        assert tower.vanishes(tower.exponential(tower.element(argument)) - expected), argument
    assert len(tower.generators) == count                      # nothing was adjoined
    # a rational multiple brings a root, not a new exponential
    half = tower.exponential(tower.element(x / 2))
    assert _kinds(tower).count(EXPONENTIAL) == 2 and _kinds(tower).count(ROOT) == 1
    assert tower.vanishes(half * half - a)
    assert tower.vanishes(tower.exponential(tower.element(3 * x / 2 + y)) - half**3 * b)
    assert tower.certified


def test_the_constant_of_a_dependent_exponential() -> None:
    # exp(x + 1) after exp(x): the constant exp(1) is a generator of its own
    tower = ElementaryTower()
    a = tower.exponential(tower.element(x))
    shifted = tower.exponential(tower.element(x + 1))
    e = tower.element(E)
    assert tower.vanishes(shifted - a * e)
    assert tower.vanishes(tower.exponential(tower.element(x + 3)) - a * e**3)
    assert tower.vanishes(tower.exponential(tower.element(x + log(2))) - a.times(Integer(2)))
    assert tower.vanishes(tower.exponential(tower.element(x + I * pi)) + a)
    assert tower.vanishes(tower.exponential(tower.element(x + I * pi / 2)) - a * tower.element(I))
    # exp(x + sqrt(2)) brings exp(sqrt(2)), independent of exp(1)
    before = _kinds(tower).count(EXPONENTIAL)
    tower.exponential(tower.element(x + sqrt(2)))
    assert _kinds(tower).count(EXPONENTIAL) == before + 1 and tower.certified


def test_constant_exponentials() -> None:
    tower = ElementaryTower()
    e = tower.element(E)
    assert tower.vanishes(tower.element(exp(2)) - e * e)
    assert tower.vanishes(tower.element(exp(Rational(1, 2)))**2 - e)
    assert tower.vanishes(tower.element(exp(Rational(-3, 2))) * tower.element(exp(Rational(3, 2))) - tower.one)
    assert tower.vanishes(tower.element(exp(sqrt(2)) * exp(sqrt(8))) - tower.element(exp(3 * sqrt(2))))
    assert tower.vanishes(tower.element(exp(sqrt(2) + sqrt(3))) - tower.element(exp(sqrt(2))) * tower.element(exp(sqrt(3))))
    assert tower.vanishes(tower.element(exp(log(3))) - tower.constant(Rational(3)))
    assert tower.vanishes(tower.element(exp(log(3) / 2)) - tower.element(sqrt(3)))
    assert tower.vanishes(tower.element(exp(2 * I * pi)) - tower.one)
    assert tower.vanishes(tower.element(exp(I * pi / 4)) - tower.element((1 + I) / sqrt(2)))
    assert tower.vanishes(tower.element(exp(pi)) - tower.element((-1)**(-I)))
    assert tower.certified


def test_an_exponential_of_a_logarithm_is_a_power() -> None:
    tower = ElementaryTower()
    assert tower.to_expr(tower.element(exp(log(x)))) == x
    assert tower.to_expr(tower.element(exp(3 * log(x)))) == x**3
    assert tower.to_expr(tower.element(exp(-log(x)))) == 1 / x
    assert tower.to_expr(tower.element(exp(log(x) / 2))) == sqrt(x)
    assert tower.to_expr(tower.element(exp(log(x) + log(y)))) == x * y          # exact for every x and y
    assert tower.to_expr(tower.element(exp(y * log(x)) - x**y)) == 0
    assert tower.to_expr(tower.element(exp(2 * log(x) + 3))) == x**2 * exp(3) or \
        tower.vanishes(tower.element(exp(2 * log(x) + 3)) - tower.element(x**2) * tower.element(E)**3)


# ---------------------------------------------------------------------------
# logarithms of numbers

def test_logarithms_of_rational_numbers() -> None:
    tower = ElementaryTower()
    two, three = tower.element(log(2)), tower.element(log(3))
    assert tower.vanishes(tower.element(log(12)) - two.times(Integer(2)) - three)
    assert tower.vanishes(tower.element(log(Rational(8, 27))) - two.times(Integer(3)) + three.times(Integer(3)))
    assert tower.vanishes(tower.element(log(1)))
    i_pi = tower.element(I * pi)
    assert tower.vanishes(tower.element(log(-1)) - i_pi)
    assert tower.vanishes(tower.element(log(Rational(-3, 2))) - three + two - i_pi)
    # one generator for each prime, and log(-1)
    assert sorted(str(g.expression) for g in tower.generators) == ['I*pi', 'log(2)', 'log(3)']
    assert tower.certified
    raises(NotElementary, lambda: tower.element(log(0)))


def test_logarithms_of_algebraic_numbers_with_a_rational_power() -> None:
    tower = ElementaryTower()
    i_pi, two = tower.element(I * pi), tower.element(log(2))
    assert tower.vanishes(tower.element(log(I)) - i_pi.times(Rational(1, 2)))
    assert tower.vanishes(tower.element(log(-I)) + i_pi.times(Rational(1, 2)))
    assert tower.vanishes(tower.element(log(sqrt(2))) - two.times(Rational(1, 2)))
    assert tower.vanishes(tower.element(log(-sqrt(2))) - two.times(Rational(1, 2)) - i_pi)
    assert tower.vanishes(tower.element(log(2**Rational(1, 3))) - two.times(Rational(1, 3)))
    assert tower.vanishes(tower.element(log(1 + I)) - two.times(Rational(1, 2)) - i_pi.times(Rational(1, 4)))
    assert tower.vanishes(tower.element(log(-1 - I)) - two.times(Rational(1, 2)) + i_pi.times(Rational(3, 4)))
    assert tower.vanishes(tower.element(log(exp(2 * I * pi / 3))) - i_pi.times(Rational(2, 3)))
    assert tower.vanishes(tower.element(log(exp(-2 * I * pi / 3))) + i_pi.times(Rational(2, 3)))
    assert tower.certified


def test_logarithms_of_algebraic_numbers_through_multiplicative_relations() -> None:
    # found by PSLQ on the moduli and the arguments, verified in the number field
    tower = ElementaryTower()
    unit = tower.element(log(1 + sqrt(2)))
    assert _kinds(tower) == [LOGARITHM]
    assert tower.vanishes(tower.element(log(3 + 2 * sqrt(2))) - unit.times(Integer(2)))      # (1 + sqrt(2))**2
    assert tower.vanishes(tower.element(log(sqrt(2) - 1)) + unit)                             # its inverse
    assert tower.vanishes(tower.element(log(7 + 5 * sqrt(2))) - unit.times(Integer(3)))
    assert tower.vanishes(tower.element(log(2 + 2 * sqrt(2))) - unit - tower.element(log(2)))
    assert _kinds(tower).count(LOGARITHM) == 2
    # Gaussian integers: (2 + I)*(2 - I) = 5 and (2 + I)**2 = 3 + 4*I
    tower = ElementaryTower()
    a = tower.element(log(2 + I))
    assert tower.vanishes(tower.element(log(2 - I)) + a - tower.element(log(5)))
    assert tower.vanishes(tower.element(log(3 + 4 * I)) - a.times(Integer(2)))
    # the turn: (-2 - I) = -(2 + I), whose logarithm is log(2 + I) - I*pi
    assert tower.vanishes(tower.element(log(-2 - I)) - a + tower.element(I * pi))
    # a lone unit is independent of the primes: certified; two unrelated ones are not proved so
    lone = ElementaryTower()
    lone.element(log(1 + sqrt(2)) + log(2) + log(3))
    assert lone.certified
    pair = ElementaryTower()
    pair.element(log(1 + sqrt(2)) + log(2 + sqrt(3)))
    assert not pair.certified


def test_moduli_of_the_conjugates() -> None:
    assert ElementaryTower._moduli_differ(1 + sqrt(2))              # 1 + sqrt(2) and 1 - sqrt(2)
    assert ElementaryTower._moduli_differ(2 + sqrt(3))
    assert not ElementaryTower._moduli_differ(2 + I)               # 2 + I and 2 - I
    assert not ElementaryTower._moduli_differ(2**Rational(1, 3))   # the three cube roots of 2


def test_logarithms_of_positive_constants() -> None:
    tower = ElementaryTower()
    e = tower.element(E)
    assert tower.vanishes(tower.element(log(E)) - tower.one)
    assert tower.vanishes(tower.element(log(E**3)) - tower.constant(Rational(3)))
    assert tower.vanishes(tower.element(log(2 * E**2)) - tower.constant(Rational(2)) - tower.element(log(2)))
    assert tower.vanishes(tower.element(log(exp(sqrt(2)) / 3)) - tower.element(sqrt(2)) + tower.element(log(3)))
    assert tower.vanishes(tower.element(log(sqrt(E))) - tower.constant(Rational(1, 2)))
    assert tower.certified and e is not None
    # log(pi) and log(1 + E) are generators whose independence is not proved
    other = ElementaryTower()
    other.element(log(pi))
    assert not other.certified
    other = ElementaryTower()
    other.element(log(1 + E))
    assert not other.certified
    assert is_zero(log(pi) - 1) is False and is_zero(log(pi) - log(pi)) is True


# ---------------------------------------------------------------------------
# logarithms of functions

def test_an_independent_logarithm_is_a_generator() -> None:
    tower = ElementaryTower()
    for argument in (x, x + 1, x**2 + y, exp(x) + 1, log(x)):
        before = _kinds(tower).count(LOGARITHM)
        tower.element(log(argument))
        assert _kinds(tower).count(LOGARITHM) == before + 1, argument
    assert tower.certified


def test_positive_factors_leave_a_logarithm() -> None:
    # log(u*w) = log(u) + log(w) for u > 0 and any w
    tower = ElementaryTower()
    lx = tower.element(log(x))
    assert tower.vanishes(tower.element(log(2 * x)) - lx - tower.element(log(2)))
    assert tower.vanishes(tower.element(log(x / 3)) - lx + tower.element(log(3)))
    assert tower.vanishes(tower.element(log(-2 * x)) - tower.element(log(-x)) - tower.element(log(2)))
    assert tower.vanishes(tower.element(log(p * x)) - tower.element(log(p)) - lx)
    assert tower.vanishes(tower.element(log(p**2 * q / x)) - tower.element(log(p)).times(Integer(2))
                          - tower.element(log(q)) - tower.element(log(1 / x)))
    assert tower.vanishes(tower.element(log((p + 1) * x)) - tower.element(log(p + 1)) - lx)
    # a negative factor gives its I*pi where the other is positive
    assert is_zero(log(-p) - log(p) - I * pi) is True
    assert is_zero(log(-p * q) - log(p) - log(q) - I * pi) is True
    assert is_zero(log(-x) - log(x) - I * pi) is False                    # x = -1: -2*I*pi
    n = Symbol('n', negative=True)
    assert is_zero(log(n) - log(-n) - I * pi) is True
    assert is_zero(log(n * p) - log(-n) - log(p) - I * pi) is True
    assert is_zero(log(n**2) - 2 * log(-n)) is True


def test_a_dependent_logarithm_where_the_arguments_are_positive() -> None:
    tower = ElementaryTower([x > 0, y > 0])
    lx, ly = tower.element(log(x)), tower.element(log(y))
    assert tower.vanishes(tower.element(log(x * y)) - lx - ly)
    assert tower.vanishes(tower.element(log(x**3 / y**2)) - lx.times(Integer(3)) + ly.times(Integer(2)))
    assert tower.vanishes(tower.element(log(sqrt(x))) - lx.times(Rational(1, 2)))
    assert tower.vanishes(tower.element(log(1 / x)) + lx)
    assert _kinds(tower).count(LOGARITHM) == 2 and tower.certified
    # log(exp(a)) = a for a real a
    tower = ElementaryTower(x > 0)
    assert tower.vanishes(tower.element(log(exp(x))) - tower.element(x))
    assert tower.vanishes(tower.element(log(exp(2 * x) * x)) - tower.element(2 * x + log(x)))
    assert tower.vanishes(tower.element(log(2 * exp(-x))) - tower.element(log(2) - x))
    assert tower.certified
    # with E as a constant factor: log(E*x) = 1 + log(x)
    assert is_zero(log(E * x) - 1 - log(x), x > 0) is True
    assert is_zero(log(x / E**2) + 2 - log(x), x > 0) is True


def test_a_dependent_logarithm_of_unknown_branch() -> None:
    tower = ElementaryTower()
    tower.element(log(x) + log(y))
    assert tower.certified
    dependent = tower.element(log(x * y))
    assert _kinds(tower).count(LOGARITHM) == 3 and not tower.certified
    assert not tower.generators[-1].certified
    assert not tower.vanishes(dependent - tower.element(log(x)) - tower.element(log(y)))
    # the same logarithm twice is the same generator, whatever its branch
    assert tower.vanishes(tower.element(log(x * y)) - dependent)
    assert is_zero(log(x * y) - log(y * x)) is True
    assert is_zero(2 * log(x * y) - log(x * y) - log(x * y)) is True


def test_logarithms_of_roots_and_powers_of_roots() -> None:
    # log(rho**k) = k*log(w)/q for 0 < k < q, whose imaginary part stays inside (-pi, pi)
    tower = ElementaryTower()
    lx = tower.element(log(x))
    assert tower.vanishes(tower.element(log(sqrt(x))) - lx.times(Rational(1, 2)))
    assert tower.vanishes(tower.element(log(x**Rational(1, 3))) - lx.times(Rational(1, 3)))
    assert tower.vanishes(tower.element(log(x**Rational(2, 3))) - lx.times(Rational(2, 3)))
    assert tower.vanishes(tower.element(log(x**Rational(5, 6))) - lx.times(Rational(5, 6)))
    assert tower.certified
    # beyond one turn it is not an identity: log(x**(3/2)) at x = -1 is -I*pi/2, not 3*I*pi/2
    assert is_zero(log(x**Rational(3, 2)) - 3 * log(x) / 2) is False
    assert is_zero(log(x**Rational(3, 2)) - 3 * log(x) / 2, x > 0) is True
    # a negative power too: sqrt(x) is never on the negative axis
    assert is_zero(log(1 / sqrt(x)) + log(x) / 2) is True
    assert is_zero(log(x**Rational(-2, 3)) + 2 * log(x) / 3) is True


def test_the_pinned_constant_of_a_complex_logarithm() -> None:
    # complex arguments on a convex real region which no argument's cut crosses
    assert is_zero(log(1 + I * x) + log(1 - I * x) - log(1 + x**2), [x > -5, x < 5]) is True
    assert is_zero(log((1 + I * x) / (1 - I * x)) - 2 * I * atan(x), [x > -5, x < 5]) is True
    assert is_zero(log(x + I) - log(x - I) - 2 * I * atan(1 / x), x > 0) is True
    assert is_zero(log(x + I) - log(x - I) - 2 * I * atan(1 / x) + 2 * I * pi, x < 0) is False
    assert is_zero(log(I * x) - log(x) - I * pi / 2, x > 0) is True
    assert is_zero(log(I * x) - log(x) - I * pi / 2, x < 0) is False       # -2*I*pi there
    assert is_zero(log(I * x) - log(x) + 3 * I * pi / 2, x < 0) is True
    assert is_zero(log(-I * x) - log(-x) - I * pi / 2, x < 0) is True
    # refused: a region which is not convex, a pole inside, a complex variable
    assert is_zero(atan(x) + atan(1 / x) - pi / 2, x**2 > 1) is False      # x = -2
    assert is_zero(atan(x) + atan(1 / x) - pi / 2, [x > -1, x < 1]) is False
    assert is_zero(atan(x) + atan(1 / x) - pi / 2) is False
    assert is_zero(atan(x) + atan(1 / x) - pi / 2, x > 0) is True
    # square roots of radicands positive on the region are positive numbers there
    inside = [x > -1, x < 1]
    assert is_zero(asin(x) - atan(x / sqrt(1 - x**2)), inside) is True
    assert is_zero(acos(x) - pi / 2 + atan(x / sqrt(1 - x**2)), inside) is True
    assert is_zero(asin(x) - 2 * atan(x / (1 + sqrt(1 - x**2))), inside) is True
    assert is_zero(asin(x) - atan(x / sqrt(1 - x**2)), x > 1) is False


# ---------------------------------------------------------------------------
# roots and their certification

def test_roots_of_numbers_are_algebraic_numbers() -> None:
    tower = ElementaryTower()
    assert tower.as_number(tower.element(sqrt(2))) == sqrt(2)
    assert tower.as_number(tower.element(exp(log(2) / 2))) == sqrt(2)
    assert tower.as_number(tower.element(2**Rational(3, 2))) == 2 * sqrt(2)
    assert tower.as_number(tower.element(sqrt(-4))) == 2 * I
    assert tower.as_number(tower.element((-8)**Rational(1, 3)) - tower.element(1 + sqrt(3) * I)) == 0   # the principal root
    assert tower.as_number(tower.element(Rational(4, 9)**Rational(1, 2))) == Rational(2, 3)
    assert not any(g.kind == ROOT for g in tower.generators)


def test_kummer_certification() -> None:
    cases = [
        (sqrt(x) + sqrt(y), True), (sqrt(x) + sqrt(x + 1), True), (sqrt(x) + x**Rational(1, 3), True),
        (sqrt(x) + sqrt(y) + sqrt(x + y) + sqrt(x - y), True), (sqrt(x * y) + sqrt(x), True),
        (sqrt(x**2 - 1) + sqrt(x - 1), True), (sqrt(2 * x) + sqrt(x), True), (sqrt(exp(x) + 1) + sqrt(x), True),
        (x**Rational(1, 3) + (x + 1)**Rational(1, 3) + sqrt(x * (x + 1)), True),
        (sqrt(x) + sqrt(y) + sqrt(x * y), False),                   # sqrt(x*y) = +-sqrt(x)*sqrt(y)
        (sqrt(x**2), False), (sqrt(x**2 - 1) + sqrt(x - 1) + sqrt(x + 1), False),
        ((x**2)**Rational(1, 4) + sqrt(x), False), (sqrt(x**3), True), (sqrt(x**3) + sqrt(x), False),
        (sqrt(1 + sqrt(x)), False),                                  # a root under a root: not certified
    ]
    for expression, expected in cases:
        tower = ElementaryTower()
        assert tower.certifies(tower.element(expression)) is expected, expression
    # the logarithm through which sqrt(x*(x + 1)) is read depends on log(x)
    # and log(x + 1): the tower is not certified, the roots are
    tower = ElementaryTower()
    u = tower.element(x**Rational(1, 3) + (x + 1)**Rational(1, 3) + sqrt(x * (x + 1)))
    assert tower.certifies(u) and not tower.certified
    # whichever of the three logarithms came last is the dependent one
    assert not tower.certifies(u + tower.element(log(x * (x + 1)) + 2 * log(x) + 3 * log(x + 1)))
    # with the signs known the dependent roots are not adjoined at all
    tower = ElementaryTower([x > 0, y > 0])
    tower.element(sqrt(x) + sqrt(y) + sqrt(x * y) + sqrt(x**2) + sqrt(x**3))
    assert _kinds(tower).count(ROOT) == 2 and tower.certified


def test_identities_between_roots() -> None:
    assert is_zero(sqrt(x)**3 - x * sqrt(x)) is True
    assert is_zero(x**Rational(1, 3) * x**Rational(1, 6) - sqrt(x)) is True
    assert is_zero((sqrt(x) + sqrt(y)) * (sqrt(x) - sqrt(y)) - x + y) is True
    assert is_zero(1 / (sqrt(x + 1) - sqrt(x)) - sqrt(x + 1) - sqrt(x)) is True
    assert is_zero(sqrt(x) / x - 1 / sqrt(x)) is True
    assert is_zero(sqrt(1 / x) - 1 / sqrt(x)) is False                     # x = -1: -I against I ... -I
    assert is_zero(sqrt(1 / x) - 1 / sqrt(x), x > 0) is True
    assert is_zero(sqrt(4 * x) - 2 * sqrt(x)) is True
    assert is_zero(sqrt(-x) - I * sqrt(x)) is False                        # x = -1
    assert is_zero(sqrt(-x) - I * sqrt(x), x > 0) is True
    assert is_zero(sqrt(-x) + I * sqrt(x), x < 0) is True
    assert is_zero(sqrt(exp(x)) - exp(x / 2)) is False                     # x = 4*I: the other root
    assert is_zero(sqrt(exp(x)) - exp(x / 2), x > 0) is True
    assert is_zero(sqrt(x + 1)**2 * sqrt(x)**4 - x**2 * (x + 1)) is True
    assert is_zero(sqrt(x) - sqrt(y)) is False and is_zero(sqrt(x) + sqrt(x + 1) - sqrt(4 * x + 2)) is False


# ---------------------------------------------------------------------------
# reading expressions

def test_every_supported_function_is_read_exactly() -> None:
    points = [{x: Rational(3, 10) + I / 7, y: Rational(-5, 4) + 2 * I}, {x: Rational(-11, 5), y: Rational(2, 3)},
              {x: Rational(1, 4), y: Rational(1, 2) - I}]
    expressions = [sin(x), cos(x), tan(x), cot(x), sec(x), csc(x), sinh(x), cosh(x), tanh(x), coth(x),
                   asin(x), acos(x), atan(x), acot(x), asinh(x), acosh(x), atanh(x),
                   x**y, 2**x, x**Rational(-3, 2), (x + y)**Rational(2, 3), exp(x)**y, E**(x * y), pi**x, x**pi,
                   log(x + y) / (1 + exp(x)), sqrt(sin(x) + 2), atan(exp(x)), log(cos(y) + 3)]
    for expression in expressions:
        found = canonical_form(expression)
        for point in points:
            left, right = _value(expression, point), _value(found, point)
            assert abs(left - right) <= 1e-9 * (1 + abs(left)), (expression, point)


def test_what_is_not_elementary() -> None:
    for expression in (gamma(x), erf(x), Abs(x), x + Float(0.5), Float(2.0) * x, log(x - x), sin(gamma(x)),
                       exp(Abs(x))):
        raises(NotElementary, lambda: canonical_form(expression))
    raises(NotElementary, lambda: ElementaryTower().element(0**x))
    raises(TypeError, lambda: untyped(canonical_form)([1, 2]))
    # zero to a positive integer power and division are left to SymPy's own evaluation
    assert canonical_form(x**0) == 1 and canonical_form(x / x) == 1


def test_the_cache_and_the_order_of_the_generators() -> None:
    tower = ElementaryTower()
    a = tower.element(exp(x) + log(x))
    count = len(tower.generators)
    assert tower.element(exp(x) + log(x)) is a
    tower.element(exp(x) * log(x)**2 + exp(2 * x) / log(x))
    assert len(tower.generators) == count
    assert [g.expression for g in tower.generators][0] == x


def test_round_trip_through_the_canonical_form() -> None:
    # the canonical form of a canonical form is itself
    for expression in (exp(x + y) + sin(x)**2, log(x**2 + 1) / sqrt(x), (sqrt(x) + 1)**3, x**y + 2**x, tan(x) + 1 / x):
        once = canonical_form(expression)
        assert is_zero(once - expression) is True
        assert is_zero(canonical_form(once) - once) is True


# ---------------------------------------------------------------------------
# the decision and its witness

def test_the_three_answers() -> None:
    assert is_zero(S.Zero) is True and is_zero(S.One) is False and is_zero(pi - pi) is True
    assert is_zero(exp(x) - exp(y)) is False
    assert is_zero(gamma(x + 1) - x * gamma(x)) is None        # true, and outside the elementary functions
    assert is_zero(erf(x) - erf(x)) is True                    # SymPy cancels it before the tower
    assert is_zero(gamma(x) - 2) is False                      # a sample point
    assert equal(exp(x)**2, exp(2 * x)) is True and equal(sqrt(x)**2, x) is True
    assert equal(log(x**2), 2 * log(x)) is False and equal(log(x**2), 2 * log(x), x > 0) is True
    assert equal(gamma(x), gamma(x + 1) / x) is None
    # assumptions as one Boolean, a list, a conjunction
    for assumptions in (x > 0, [x > 0], (x > 0) & (y > 0), [x > 0, y > 0]):
        assert is_zero(log(x**3) - 3 * log(x), assumptions) is True


def test_an_uncertified_nonzero_needs_a_witness() -> None:
    tower = ElementaryTower()
    u = tower.element(log(x**2) - 2 * log(x))
    assert not tower.vanishes(u) and not tower.certified
    assert _witness(as_expr(log(x**2) - 2 * log(x)), None)                   # x = -1 among the samples
    assert not _witness(as_expr(log(p**2) - 2 * log(p)), None)               # none for a positive symbol
    assert not _witness(as_expr(log(x**2) - 2 * log(x)), [x > 0])
    assert _witness(as_expr(log(exp(x)) - x), None)                          # a complex point
    assert not _witness(as_expr(log(exp(x)) - x), [x > 0])
    r = Symbol('r', real=True)
    assert _witness(as_expr(sqrt(r**2) - r + exp(r) - exp(r)), None) or is_zero(sqrt(r**2) - r) is False
    assert _witness(as_expr(atan(x) + atan(1 / x) - pi / 2), [x > -1, x < 1])  # the other sign inside the interval
    # without numerical checks there is no witness: None, never a guess
    with configure(numerical_checks=False):
        assert not _witness(as_expr(log(x**2) - 2 * log(x)), None)
        assert is_zero(log(x**2) - 2 * log(x)) is None
        assert is_zero(log(x**2) - 2 * log(x), x > 0) is True               # a proof needs no numbers
        assert is_zero(exp(x) - x - 1) is False                             # nor does a certified tower


def test_symbol_flags_are_part_of_the_region() -> None:
    r, n = Symbol('r', real=True), Symbol('n', negative=True)
    assert is_zero(log(p * q) - log(p) - log(q)) is True
    assert is_zero(sqrt(p**2) - p) is True and is_zero(sqrt(n**2) + n) is True
    assert is_zero(log(exp(r)) - r) is True
    assert is_zero(sqrt(p * q) - sqrt(p) * sqrt(q)) is True
    assert is_zero(sqrt(r**2) - r) is False
    assert is_zero((p**x)**y - p**(x * y)) is False                          # y complex: (p**x)**y is another branch
    assert is_zero((p**r)**y - p**(r * y)) is True
    assert is_zero(atan(p) + atan(1 / p) - pi / 2) is True
    assert is_zero(atan(n) + atan(1 / n) + pi / 2) is True


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


def test_a_constant_written_with_functions_is_not_a_certified_generator() -> None:
    # the bug: exp((log(x*y) - log(x) - log(y))/2), which is 1 or -1, was
    # adjoined as an exponential independent of everything
    tower = ElementaryTower()
    tower.element(log(x) + log(y))
    dependent = tower.element(log(x * y))
    half = tower.exponential((dependent - tower.element(log(x)) - tower.element(log(y))).times(Rational(1, 2)))
    assert not tower.certifies(half)
    # and the root of x*y is read from its own logarithm: a root, with its relation
    tower = ElementaryTower()
    u = tower.element(sqrt(x) + sqrt(y) + sqrt(x * y))
    assert _kinds(tower).count(ROOT) == 3 and _kinds(tower).count(EXPONENTIAL) == 0
    assert not tower.certifies(u)
    assert tower.vanishes(tower.element(sqrt(x * y))**2 - tower.element(x * y))


def test_numbers_which_sympy_does_not_convert() -> None:
    # SymPy 1.14: QQ.algebraic_field(w).from_sympy(2*w) fails for w = (-1)**(1/3),
    # and QQ.algebraic_field(2*I, 2*w) raises NotInvertible
    tower = ElementaryTower()
    a = tower.element((-8)**Rational(1, 3))
    assert tower.as_number(a - tower.element(1 + sqrt(3) * I)) == 0
    tower = ElementaryTower()
    b = tower.element(sqrt(-4) + (-8)**Rational(1, 3))
    assert tower.as_number(b - tower.element(2 * I + 1 + sqrt(3) * I)) == 0
    assert is_zero((-8)**Rational(1, 3) - 1 - sqrt(3) * I) is True
    assert is_zero((-27)**Rational(1, 3) + 3) is False                       # the principal root is not -3


def test_the_second_opinion_of_numerically_equal() -> None:
    # agreement at three sample points is evidence; two expressions which
    # the structure theorem proves different are different
    from sympy_extras.integrals.conditions import _provably_different, numerically_equal
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
    from sympy_extras.integrals.conditions import _provably_different, numerically_equal
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
