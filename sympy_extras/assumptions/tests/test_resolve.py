from __future__ import annotations

from sympy import (S, Q, Eq, Ne, And, Or, Not, Implies, symbols, Interval,
    CRootOf, sin, true, false)
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, x, y, z

from sympy_extras.assumptions import resolve, ForAll, Exists, element
from sympy_extras.polys.cad.tests.test_qe import _equivalent


def test_resolve_closed() -> None:
    assert resolve(ForAll(x, x**2 >= 0)) is true
    assert resolve(ForAll(x, x**2 > 0)) is false
    assert resolve(Exists(x, x**2 < 0)) is false
    assert resolve(Exists(x, Eq(x**2, 2))) is true
    assert resolve(ForAll(x, Exists(y, y > x))) is true
    assert resolve(Exists(y, ForAll(x, y > x))) is false
    assert resolve(ForAll([x, y], x**2 + y**2 >= 2*x*y)) is true
    assert resolve(ForAll([a, b], Exists(x, Eq(x**3 + a*x + b, 0)))) is true
    assert resolve(ForAll([a, b], Exists(x, Eq(x**2 + a*x + b, 0)))) is false
    assert resolve(~ForAll(x, x**2 > 0)) is true
    assert resolve(Not(Exists(x, x**2 < 0))) is true
    assert resolve(ForAll(x, x > 0) | Exists(x, x < 0)) is true
    assert resolve(ForAll(x, x > 0) & Exists(x, x < 0)) is false
    assert resolve(Implies(ForAll(x, x**2 >= 0), Exists(x, x > 5))) is true
    assert resolve(True) is true
    assert resolve(2 > 1) is true
    assert resolve(Eq(S(2), 3)) is false


def test_resolve_one_free_variable() -> None:
    assert resolve(ForAll(x, x**2 + b*x + 1 > 0)) == And(b > -2, b < 2)
    assert resolve(Exists(y, Eq(x**2 + y**2, 1))) == And(x >= -1, x <= 1)
    assert resolve(Exists(y, Eq(y**2, x))) == (x >= 0)
    assert resolve(Exists(y, Eq(y**2, x) & (y > 1))) == (x > 1)
    assert resolve(ForAll(y, x**2 + y**2 > 0)) == Or(x < 0, x > 0)
    assert resolve(Exists(y, Eq(x**2 + y**2, 1) & (y > x))) == And(x >= -1, x < CRootOf(2*x**2 - 1, 1))
    assert resolve(x**2 > 2) == Or(x < CRootOf(x**2 - 2, 0), x > CRootOf(x**2 - 2, 1))
    assert resolve(x**2 >= 0) is true
    assert resolve(x**2 < 0) is false
    assert resolve(Or(x > 1, x > 2)) == (x > 1)
    assert resolve(x**3 - x > 0) == Or(And(x > -1, x < 0), x > 1)
    # the quantified variable need not appear
    assert resolve(ForAll(y, x > 1)) == (x > 1)


def test_resolve_several_free_variables() -> None:
    assert resolve(ForAll(x, x**2 + b*x + c > 0)) == (b**2 < 4*c)
    assert resolve(Exists(x, Eq(x**2 + a*x + b, 0))) == (a**2 >= 4*b)
    r = resolve(Exists(x, Eq(a*x**2 + b*x + c, 0)))
    ref = And(4*a*c - b**2 <= 0, Or(Eq(c, 0), Ne(a, 0), 4*a*c - b**2 < 0))
    assert _equivalent(r, ref, [a, b, c])
    r = resolve(Exists(x, Eq(a*x**2 + b*x + c, 0) & (a > 0)))
    assert _equivalent(r, And(a > 0, 4*a*c - b**2 <= 0), [a, b, c])
    assert r == And(a > 0, 4*a*c - b**2 <= 0)
    assert resolve((x**2 + y**2 < 1) & (x > y)) == And(x - y > 0, x**2 + y**2 - 1 < 0)
    assert resolve(x*y > 0) == Or(And(x < 0, y < 0), And(x > 0, y > 0))
    # quadratic in z: virtual substitution instead of the decomposition
    r = resolve(Exists(z, Eq(z**2, x) & (z > y)))
    assert _equivalent(r, And(x >= 0, Or(y < 0, y**2 < x)), [x, y])


def test_resolve_assumptions() -> None:
    assert resolve(Exists(y, Eq(y**2, x)), assumptions=x > 1) is true
    assert resolve(Exists(y, Eq(y**2, x)), assumptions=x < 0) is false
    assert resolve(Exists(y, Eq(y**2, x) & (y > 1)), assumptions=x > 0) == (x > 1)
    assert resolve(ForAll(x, x**2 + b*x + 1 > 0), assumptions=b > 0) == (b < 2)
    assert resolve(ForAll(x, x**2 + b*x + 1 > 0), assumptions=b > 3) is false


def test_resolve_predicates() -> None:
    assert resolve(ForAll(x, Implies(Q.positive(x), x**3 > 0))) is true
    assert resolve(Exists(x, Q.negative(x) & element(x, S.Reals) & (x**2 < 1))) is true
    assert resolve(Exists(x, element(x, Interval(0, 1)) & (x > y))) == (y < 1)


def test_resolve_errors() -> None:
    raises(NotImplementedError, lambda: resolve(ForAll(x, x > 0), domain=S.Rationals))
    raises(ValueError, lambda: resolve(ForAll(x, sin(x) > 0)))
    raises(ValueError, lambda: resolve(ForAll(x, element(x, S.Integers))))
    raises(ValueError, lambda: resolve(Exists(x, x > symbols('p'))) if False else resolve(Exists(x, symbols('p'))))
