"""The native virtual substitution on objects of the core, checked at
random rational points against the native ``resolve`` of the specialised
formulas (whose closed formulas the cylindrical algebraic decomposition
decides) and, for the linear case, against a brute force search of the
candidate values (the SymPy face is tested in test_virtual_substitution.py)."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy_extras.core import Basic, Eq, Ne, Rational, Symbol
from sympy_extras.core.logic import (Boolean, BooleanFalse, BooleanTrue, Equivalent, ITE, Implies, Not, Xor, and_,
    false, or_, to_boolean, true)
from sympy_extras.core.numbers import number_from_fraction
from sympy_extras.assumptions.quantifiers import Exists
from sympy_extras.assumptions.resolve import resolve
from sympy_extras.polys.virtual_substitution import (eliminate_linear, eliminate_quadratic, is_linear_in,
    is_quadratic_in, linear_quantifier_elimination, to_nnf, virtual_substitution_elimination, with_sides)
from sympy_extras._testing import untyped
from sympy.testing.pytest import raises

x, y, z, a, b, c = (Symbol(s) for s in 'xyzabc')


def _point(rng: random.Random, symbols: list[Symbol]) -> dict[Basic, Basic]:
    return {s: number_from_fraction(Fraction(rng.randint(-4, 4), rng.choice([1, 2, 3]))) for s in symbols}


def _parameters(f: Boolean) -> list[Symbol]:
    return sorted((s for s in f.free_symbols if s != x), key=lambda s: s.name)


def _truth(f: Boolean) -> bool:
    assert isinstance(f, (BooleanTrue, BooleanFalse)), f
    return isinstance(f, BooleanTrue)


def test_is_linear_in() -> None:
    assert is_linear_in(and_(2*x + y > 0, x*y < 1), x)
    assert not is_linear_in(x**2 + y > 0, x)
    assert is_linear_in(y**2 > 0, x)
    assert not is_linear_in(Eq(x*y*x, 1), x)
    # a coefficient outside the domains of the native Poly (QQ(a), EX)
    assert is_linear_in(and_(x/a > 1, x*Rational(1, 2) < b), x)
    assert is_quadratic_in(and_(x**2 + y*x > 0, x < 1), x) and not is_quadratic_in(x**3 > y, x)


def test_eliminate_linear() -> None:
    assert eliminate_linear(and_(x > y, x < 1), x) == (y < 1)
    assert eliminate_linear(Eq(a*x, 1), x) == Ne(a, 0)
    assert eliminate_linear(and_(x > 0, x < 0), x) is false
    assert eliminate_linear(or_(x > y, x < z), x) is true
    assert eliminate_linear(and_(x >= y, x <= z), x) == (y <= z)
    assert eliminate_linear(y > 0, x) == (y > 0)
    # a pinned value: x = 2 substituted
    assert eliminate_linear(and_(Eq(x, 2), x > y), x) == (y < 2)
    raises(ValueError, lambda: eliminate_linear(x**2 > 1, x))
    raises(TypeError, lambda: untyped(eliminate_linear)(x + 1, x))


def test_eliminate_linear_at_random_points() -> None:
    formulas: list[Boolean] = [
        and_(x < 2, a*x + b > 0), and_(x > 0, Eq(a*x + b, 0)), and_(x > 2, x*y < 1),
        or_(and_(x > a, x < b), Eq(x, 3)), and_(a*x > b, x <= 0, Ne(x, -1)),
        and_(x - a > 0, b - x >= 0, x*y - 1 < 0), Not(or_(x > a, x < b)), Implies(x > a, x > b),
        Equivalent(x > a, x < b), Xor(x > a, x > b), ITE(x > 0, x < a, x > b),
        and_(a*x + b >= 0, a*x + b <= 0), and_(x > 0, x < a, a*x + y > 0)]
    rng = random.Random(7)
    for f in formulas:
        result = eliminate_linear(f, x)
        assert not result.has(x), (f, result)
        parameters = _parameters(f)
        for _ in range(6):
            point = _point(rng, parameters)
            expected = resolve(Exists(x, to_boolean(f.subs(point))))
            actual = to_boolean(result.subs(point))
            assert _truth(actual) == _truth(expected), (f, point, result)


def test_linear_quantifier_elimination() -> None:
    assert linear_quantifier_elimination(and_(x > y, y > z), [('exists', x)]) == (y > z, [])
    assert linear_quantifier_elimination(and_(x > y, y > z), [('forall', z), ('exists', x)]) == (false, [])
    assert linear_quantifier_elimination(and_(x > y, y > z), [('exists', z), ('exists', x)]) == (true, [])
    f, prefix = linear_quantifier_elimination(and_(x**2 > y, y > z), [('exists', x), ('exists', z)])
    assert f == (x**2 > y) and prefix == [('exists', x)]


def test_eliminate_quadratic() -> None:
    assert eliminate_quadratic(Eq(x**2 + b*x + c, 0), x) == (b**2 >= 4*c)
    assert eliminate_quadratic(and_(x**2 < a, x > 1), x) == (a > 1)
    assert eliminate_quadratic(x**2 < 0, x) is false
    assert eliminate_quadratic(x**2 + 1 > 0, x) is true
    raises(ValueError, lambda: eliminate_quadratic(x**3 > y, x))
    formulas: list[Boolean] = [
        Eq(a*x**2 + b*x + c, 0), and_(x**2 + b*x + c > 0, x > 0), and_(x**2 < y, x > y - 1),
        or_(a*x**2 + 1 <= 0, x < a), and_(Ne(x**2 - a, 0), x >= 0, x <= 1),
        and_(Eq(x**2, a), x < b), and_(x**2 >= a, x**2 <= b), and_(a*x**2 + b*x + 1 >= 0, x < 0)]
    rng = random.Random(5)
    for f in formulas:
        result = eliminate_quadratic(f, x)
        assert not result.has(x), (f, result)
        parameters = _parameters(f)
        for _ in range(6):
            point = _point(rng, parameters)
            expected = resolve(Exists(x, to_boolean(f.subs(point))))
            actual = to_boolean(result.subs(point))
            assert _truth(actual) == _truth(expected), (f, point, result)


def test_virtual_substitution_elimination() -> None:
    assert virtual_substitution_elimination(Eq(x**2 + b*x + c, 0), [('exists', x)]) == (b**2 >= 4*c, [])
    assert virtual_substitution_elimination(x**2 + b*x + c > 0, [('forall', x)]) == (b**2 < 4*c, [])
    assert virtual_substitution_elimination(x**3 > y, [('exists', x)]) == (x**3 > y, [('exists', x)])
    # the quadratic case is refused beyond max_free free variables
    assert virtual_substitution_elimination(x**2 + b*x + c > 0, [('forall', x)], max_free=1)[1] == [('forall', x)]


def test_with_sides_and_nnf() -> None:
    assert str(with_sides(and_(x - y > 0, x + 1 < 0))) == '(x > y) & (x < -1)'
    assert with_sides(Eq(x - 1, 0)) == Eq(x, 1)
    assert to_nnf(Implies(x > 0, Not(y < 1))) == or_(x <= 0, y >= 1)
    assert to_nnf(Not(Xor(x > 0, y > 0))) == and_(or_(x > 0, y <= 0), or_(x <= 0, y > 0))
