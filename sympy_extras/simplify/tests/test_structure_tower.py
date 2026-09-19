"""Tests of the machinery of the tower, algorithm by algorithm: the field
arithmetic and the lifting of elements, the normal form, the derivations,
the linear algebra over the rationals, the number field, the questions
about the region."""
from __future__ import annotations

from sympy import (E, I, Integer, Rational, S, Symbol, cos, diff, exp, log, pi, simplify, sin, sqrt, symbols)
from sympy.core.expr import Expr
from sympy.polys.domains import QQ
from sympy.testing.pytest import raises

from sympy_extras._typing import as_expr
from sympy_extras.simplify import Element, ElementaryTower
from sympy_extras.simplify.structure import EXPONENTIAL, LOGARITHM, ROOT, VARIABLE, as_rational

x, y, z = symbols('x y z')


def _same(tower: ElementaryTower, a: Element, expression: Expr) -> bool:
    """Whether ``a`` is ``expression``, by the numerical values at two complex points."""
    found = tower.to_expr(a)
    for point in ({x: Rational(7, 5) + I / 3, y: Rational(-2, 3) + I, z: Rational(5, 4)},
                  {x: Rational(1, 7) - 2 * I, y: Rational(9, 4), z: Rational(-1, 3) + I / 2}):
        left = complex(as_expr(found.xreplace(point)).evalf(30))
        right = complex(as_expr(expression.xreplace(point)).evalf(30))
        if abs(left - right) > 1e-10 * (1 + abs(left)):
            return False
    return True


# ---------------------------------------------------------------------------
# the field arithmetic

def test_field_operations() -> None:
    tower = ElementaryTower()
    a, b = tower.element(x), tower.element(y)
    assert tower.to_expr(a + b) == x + y and tower.to_expr(a - b) == x - y
    assert tower.to_expr(a * b) == x * y and tower.to_expr(a / b) == x / y
    assert tower.to_expr(-a) == -x
    assert tower.to_expr(a**3) == x**3 and tower.to_expr(a**-2) == x**-2 and tower.to_expr(a**0) == 1
    assert tower.to_expr(a.times(Rational(3, 7))) == 3 * x / 7
    assert tower.to_expr((a + b) * (a - b) - a * a + b * b) == 0
    # fractions are cancelled, the denominator monic
    u = (a * a - b * b) / (a.times(Integer(2)) - b.times(Integer(2)))
    assert tower.to_expr(u) == (x + y) / 2
    assert u.den == tower.ring.one
    v = tower.one / (a.times(Integer(3)) + tower.one)
    assert v.den.LC == QQ.one
    assert tower.to_expr(tower.zero) == 0 and tower.to_expr(tower.one) == 1
    assert tower.to_expr(tower.constant(Rational(-5, 3))) == Rational(-5, 3)
    assert repr(a) == 'Element(x)'


def test_division_by_zero() -> None:
    tower = ElementaryTower()
    a = tower.element(x)
    raises(ZeroDivisionError, lambda: a / tower.zero)
    raises(ZeroDivisionError, lambda: a / (a - a))
    raises(ZeroDivisionError, lambda: tower.zero**-1)
    raises(ZeroDivisionError, lambda: tower._fraction(tower.ring.one, tower.ring.zero))


def test_elements_are_lifted_when_the_tower_grows() -> None:
    # an element keeps the ring of its time: a new generator and a larger
    # number field both change the ring, and the old elements still combine
    tower = ElementaryTower()
    a = tower.element(x)
    ring_before = tower.ring
    b = tower.element(exp(x))
    assert tower.ring != ring_before and a.num.ring == ring_before
    assert tower.to_expr(a * b) == x * exp(x)
    c = tower.element(sqrt(2) * x)                  # the number field grows
    assert not tower.domain.is_QQ
    assert tower.to_expr(a + b + c) == x + sqrt(2) * x + exp(x) or _same(tower, a + b + c, x + sqrt(2) * x + exp(x))
    d = tower.element(I * y)                         # and grows again
    assert _same(tower, a * c + d * b, sqrt(2) * x**2 + I * y * exp(x))
    assert tower.vanishes(c * c - a * a.times(Integer(2)))


def test_elements_of_another_tower_are_refused() -> None:
    first, second = ElementaryTower(), ElementaryTower()
    a, b = first.element(x), second.element(x)
    raises(ValueError, lambda: a + b)
    raises(ValueError, lambda: a * b)
    raises(ValueError, lambda: first.normal(b))


# ---------------------------------------------------------------------------
# the normal form

def test_the_powers_of_a_root_are_reduced() -> None:
    tower = ElementaryTower()
    r = tower.element(sqrt(x))
    assert tower.vanishes(r * r - tower.element(x))
    assert tower.to_expr(r**5) == x**2 * sqrt(x)
    assert tower.to_expr(r**-2) == 1 / x
    c = tower.element(x**Rational(1, 3))
    assert tower.to_expr(c**7 * r**3) == x**3 * sqrt(x) * x**Rational(1, 3)
    # a root whose radicand is a fraction: the ring generator is d*rho
    tower = ElementaryTower()
    q = tower.element(sqrt((x + 1) / (x - 1)))
    assert tower.vanishes(q * q - tower.element((x + 1) / (x - 1)))
    root = [g for g in tower.generators if g.kind == ROOT][0]
    assert root.scale is not None and tower.to_expr(root.scale) == x - 1
    assert root.radicand is not None and tower.to_expr(root.radicand) == x**2 - 1       # n*d**(q - 1)
    assert _same(tower, q, sqrt((x + 1) / (x - 1)))


def test_a_root_of_a_larger_index_replaces_the_smaller() -> None:
    tower = ElementaryTower()
    two = tower.element(sqrt(x))
    assert [g.index for g in tower.generators if g.kind == ROOT] == [2]
    three = tower.element(x**Rational(1, 3))
    roots = [g for g in tower.generators if g.kind == ROOT]
    assert [g.index for g in roots] == [2, 6] and roots[0].eliminated is not None and roots[1].eliminated is None
    six = tower.element(x**Rational(1, 6))
    assert len([g for g in tower.generators if g.kind == ROOT]) == 2        # the master is reused
    assert tower.vanishes(six**3 - two) and tower.vanishes(six**2 - three)
    assert tower.vanishes(two * three - tower.element(x**Rational(5, 6)))
    # the same for a fraction, where the eliminated root is a fraction of the new one
    tower = ElementaryTower()
    a = tower.element(sqrt(x / (y + 1)))
    b = tower.element((x / (y + 1))**Rational(1, 4))
    assert tower.vanishes(b * b - a)
    assert _same(tower, a + b, sqrt(x / (y + 1)) + (x / (y + 1))**Rational(1, 4))
    assert tower.certified


def test_normal_form_of_zero_and_of_a_vanishing_denominator() -> None:
    tower = ElementaryTower()
    r = tower.element(sqrt(x))
    a = tower.element(x)
    assert tower.vanishes(tower.zero) and not tower.vanishes(tower.one)
    assert tower.vanishes(r * r - a)
    # a denominator which the relation sigma**2 == x reduces to zero
    sigma = tower.ring.gens[[g.kind for g in tower.generators].index(ROOT)]
    variable = tower.ring.gens[tower._variables[x]]
    raises(ZeroDivisionError, lambda: tower.normal(Element(tower, tower.ring.one, sigma**2 - variable)))


# ---------------------------------------------------------------------------
# the derivations

def test_derivatives_of_every_kind_of_generator() -> None:
    expressions = [x**3 * y, exp(x * y), log(x**2 + y), sqrt(x + y), exp(x) * log(x) / sqrt(x),
                   (x + exp(2 * x))**Rational(1, 3), log(log(x)), exp(exp(x) * y), sin(x * y) * cos(x),
                   sqrt(x / (y + 1)), x**y]
    for expression in expressions:
        tower = ElementaryTower()
        a = tower.element(expression)
        for symbol in (x, y):
            position = tower._variables.get(symbol)
            if position is None:
                continue
            assert _same(tower, tower.derivative(a, position), as_expr(diff(expression, symbol))), (expression, symbol)


def test_constants_have_zero_derivatives() -> None:
    tower = ElementaryTower()
    k = tower.element(pi + E + log(2) + sqrt(3))
    a = tower.element(x * pi + exp(x + log(2)))
    position = tower._variables[x]
    assert tower.vanishes(tower.derivative(k, position)) and tower.is_constant(k)
    assert not tower.is_constant(a)
    assert _same(tower, tower.derivative(a, position), pi + 2 * exp(x))
    assert tower.is_constant(tower.element(exp(x) * exp(-x)))
    assert tower.is_constant(tower.element(sin(x)**2 + cos(x)**2))


def test_formal_derivatives() -> None:
    # the partial derivative with respect to an independent generator,
    # which the roots follow through their relations
    tower = ElementaryTower()
    theta = tower.element(exp(x))
    a = tower.element(x * exp(x)**2 + sqrt(exp(x) + 1))
    position = [i for i, g in enumerate(tower.generators) if g.kind == EXPONENTIAL][0]
    found = tower._formal_derivative(a, position)
    # d/dtheta (x*theta**2 + sqrt(theta + 1)) = 2*x*theta + 1/(2*sqrt(theta + 1))
    assert _same(tower, found, 2 * x * exp(x) + 1 / (2 * sqrt(exp(x) + 1)))
    assert tower.vanishes(tower._formal_derivative(tower.element(x**2), position))
    assert tower.vanishes(tower._formal_derivative(theta, position) - tower.one)


# ---------------------------------------------------------------------------
# linear algebra over the rationals

def test_rational_combinations() -> None:
    tower = ElementaryTower()
    a, b, c = tower.element(x), tower.element(y), tower.element(exp(x))
    target = a.times(Rational(2, 3)) - b.times(Integer(5)) + c
    assert tower._solve([(target, [a, b, c])]) == [Rational(2, 3), -5, 1]
    assert tower._solve([(target, [a, b])]) is None
    assert tower._solve([(tower.zero, [a, b])]) == [0, 0]
    assert tower._solve([(tower.zero, [])]) == [] and tower._solve([(a, [])]) is None
    assert tower._solve([]) == []
    # fractions: x/(y + 1) = 3*(x/(3*y + 3))
    u = tower.element(x / (y + 1))
    v = tower.element(x / (3 * y + 3))
    assert tower._solve([(u, [v, a])]) == [3, 0]
    # a dependent basis: a free unknown is set to zero
    found = tower._solve([(a, [a, a.times(Integer(2))])])
    assert found is not None and found[0] + 2 * found[1] == 1
    # several equations at once
    assert tower._solve([(a, [a, b]), (b, [a, b])]) is None
    assert tower._solve([(a + b, [a, b]), (a.times(Integer(2)) + b.times(Integer(2)), [a.times(Integer(2)), b.times(Integer(2))])]) == [1, 1]


def test_rational_combinations_over_a_number_field() -> None:
    # the unknowns are rational: sqrt(2)*x is not a rational multiple of x
    tower = ElementaryTower()
    a = tower.element(x)
    b = tower.element(sqrt(2) * x)
    assert tower._solve([(b, [a])]) is None
    assert tower._solve([(b + a.times(Integer(3)), [a, b])]) == [3, 1]
    c = tower.element(I * x + sqrt(2) * y)
    assert tower._solve([(c, [a, b])]) is None
    assert tower._solve([(c.times(Rational(1, 2)), [c])]) == [Rational(1, 2)]
    assert tower._field_degree() == 4
    assert len(tower._components(tower.domain.from_sympy(sqrt(2) * I + 1))) == 4
    # with roots: sqrt(x)**3 = x*sqrt(x)
    tower = ElementaryTower()
    r = tower.element(sqrt(x))
    assert tower._solve([(r**3, [r * tower.element(x), r])]) == [1, 0]


# ---------------------------------------------------------------------------
# the number field

def test_algebraic_numbers() -> None:
    tower = ElementaryTower()
    assert tower.domain.is_QQ and tower._field_degree() == 1
    a = tower._number(sqrt(2))
    assert tower._field_degree() == 2 and tower.as_number(a) == sqrt(2)
    b = tower._number(sqrt(8))                                  # in the field already
    assert tower._field_degree() == 2 and tower.vanishes(b - a.times(Integer(2)))
    c = tower._number(I)
    assert tower._field_degree() == 4 and tower.as_number(c * c) == -1
    assert tower.as_number(a * a) == 2                        # the old element in the new field
    assert tower.as_number(tower.element(x)) is None
    assert tower.as_number(tower.element(exp(1))) is None
    assert tower.as_number(tower.constant(Rational(3, 4))) == Rational(3, 4)
    assert tower.as_number(tower.zero) == 0
    other = ElementaryTower()
    d = other.element(2**Rational(1, 3) * I + exp(I * pi / 3))
    number = other.as_number(d)
    assert number is not None and other._field_degree() in (6, 12)
    assert abs(complex(number.evalf(30)) - complex((2**Rational(1, 3) * I + exp(I * pi / 3)).evalf(30))) < 1e-20


def test_the_number_field_is_not_allowed_to_explode() -> None:
    # the field grows by primitive elements: sqrt(2), I, 2**(1/3) and
    # exp(I*pi/5) together have degree 48, which SymPy does not finish;
    # the tower refuses, and is_zero answers from a sample point
    from sympy_extras.simplify import NotElementary, is_zero
    tower = ElementaryTower()
    tower.element(sqrt(2) + I)
    raises(NotElementary, lambda: tower.element(2**Rational(1, 3) + exp(I * pi / 5)))
    assert is_zero(sqrt(2) + I + 2**Rational(1, 3) + exp(I * pi / 5)) is False


def test_as_rational() -> None:
    assert as_rational(Rational(3, 4)) == Rational(3, 4) and as_rational(5) == 5
    raises(TypeError, lambda: as_rational(sqrt(2)))
    raises(TypeError, lambda: as_rational(x))


# ---------------------------------------------------------------------------
# questions about the region

def test_signs_on_the_region() -> None:
    p, n, r = Symbol('p', positive=True), Symbol('n', negative=True), Symbol('r', real=True)
    tower = ElementaryTower()
    assert tower._positive(tower.element(p)) and tower._negative(tower.element(n))
    assert not tower._positive(tower.element(r)) and not tower._negative(tower.element(r))
    assert tower._positive(tower.element(p**2 + 1)) and tower._positive(tower.element(r**2 + 1))
    assert not tower._positive(tower.element(x)) and not tower._positive(tower.element(x**2 + 1))
    assert tower._real(tower.element(r)) and tower._real(tower.element(p * n)) and not tower._real(tower.element(x))
    assert tower._positive(tower.constant(Rational(1, 2))) and tower._negative(tower._number(-sqrt(2)))
    assert not tower._positive(tower._number(I)) and not tower._negative(tower._number(I))
    tower = ElementaryTower([x > 1, y < 0])
    assert tower._positive(tower.element(x)) and tower._positive(tower.element(x - 1))
    assert tower._negative(tower.element(y)) and tower._negative(tower.element(x * y))
    assert not tower._positive(tower.element(x - 2)) and not tower._negative(tower.element(x - 2))
    assert tower._real(tower.element(x + y)) and not tower._real(tower.element(z))
    # a single Boolean is accepted as well as a list
    assert ElementaryTower(x > 0)._facts() == [x > 0] and ElementaryTower()._facts() == []
    assert ElementaryTower([x > 0, y > 0])._facts() == [x > 0, y > 0]


def test_a_value_written_with_I_is_not_asked_about() -> None:
    # SymPy refuses to compare a number it knows not to be real (p - I > 0
    # raises TypeError), and a sign is a property of real values: the
    # package's ask once found exp(-I*x) > 0 on an interval
    p = Symbol('p', positive=True)
    tower = ElementaryTower([x > -1, x < 1])
    assert tower._sign(as_expr(p - I)) == 0
    assert tower._sign(exp(-I * x)) == 0 and tower._sign(exp(I * x) + 2) == 0
    assert tower._sign(exp(-x)) == 1 and tower._sign(-exp(x) - 1) == -1
    assert not tower._positive(tower.element(exp(-I * x)))


def test_convex_regions() -> None:
    assert ElementaryTower()._convex_region()
    assert ElementaryTower(x > 0)._convex_region()
    assert ElementaryTower([x > 0, y < 3, x + y <= 5])._convex_region()
    assert ElementaryTower((x > 0) & (y > 0))._convex_region()
    assert not ElementaryTower(x**2 > 1)._convex_region()                # two half lines
    assert not ElementaryTower([x > 0, x * y < 1])._convex_region()
    assert not ElementaryTower((x > 0) | (x < -1))._convex_region()
    assert not ElementaryTower(S.true & (exp(x) > 2))._convex_region()
    from sympy import Ne
    assert not ElementaryTower(Ne(x, 0))._convex_region()


def test_generator_kinds() -> None:
    tower = ElementaryTower()
    tower.element(x + exp(x) + log(x) + sqrt(x))
    kinds = [g.kind for g in tower.generators]
    assert kinds.count(VARIABLE) == 1 and kinds.count(EXPONENTIAL) == 1
    assert kinds.count(LOGARITHM) == 1 and kinds.count(ROOT) == 1
    assert repr(tower.generators[0]) == 'Generator(variable, x)'
    # the expression of a generator is its meaning
    for g in tower.generators:
        assert g.expression in (x, exp(x), log(x), sqrt(x))
    assert simplify(tower.to_expr(tower.element(x + exp(x) + log(x) + sqrt(x))) - (x + exp(x) + log(x) + sqrt(x))) == 0
