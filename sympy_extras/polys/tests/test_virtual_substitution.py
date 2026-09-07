from __future__ import annotations

from sympy import S, Eq, Ne, Or, And, Equivalent, true, false
from sympy.abc import x, y, z, a, b
from sympy.testing.pytest import raises

from sympy_extras.polys.virtual_substitution import (is_linear_in, eliminate_linear,
    linear_quantifier_elimination)
from sympy_extras.assumptions import tautology, resolve, Exists, ForAll


def _equivalent(f: object, g: object) -> bool:
    return tautology(Equivalent(f, g), domain=S.Reals) is True


def test_is_linear_in() -> None:
    assert is_linear_in((2*x + y > 0) & (x*y < 1), x)
    assert not is_linear_in(x**2 + y > 0, x)
    assert is_linear_in(y**2 > 0, x)
    assert not is_linear_in(Eq(x*y*x, 1), x)


def test_eliminate_linear() -> None:
    assert eliminate_linear((x > y) & (x < 1), x) == (y < 1)
    assert eliminate_linear(Eq(a*x, 1), x) == Ne(a, 0)
    assert eliminate_linear((x > 0) & (x < 0), x) is false
    assert eliminate_linear(Or(x > y, x < z), x) is true
    assert eliminate_linear((x >= y) & (x <= z), x) == (y <= z)
    assert eliminate_linear(y > 0, x) == (y > 0)
    raises(ValueError, lambda: eliminate_linear(x**2 > 1, x))
    # against the cylindrical algebraic decomposition
    for f in [(x < 2) & (a*x + b > 0), (x > 0) & Eq(a*x + b, 0), (x > 2) & (x*y < 1),
              Or(And(x > a, x < b), Eq(x, 3)), (a*x > b) & (x <= 0) & Ne(x, -1),
              (x - a > 0) & (b - x >= 0) & (x*y - 1 < 0)]:
        assert _equivalent(eliminate_linear(f, x), resolve(Exists(x, f)))


def test_linear_quantifier_elimination() -> None:
    assert linear_quantifier_elimination((x > y) & (y > z), [('exists', x)]) == (y > z, [])
    assert linear_quantifier_elimination((x > y) & (y > z), [('forall', z), ('exists', x)]) == (false, [])
    assert linear_quantifier_elimination((x > y) & (y > z), [('exists', z), ('exists', x)]) == (true, [])
    f, prefix = linear_quantifier_elimination((x**2 > y) & (y > z), [('exists', x), ('exists', z)])
    assert f == (x**2 > y) and prefix == [('exists', x)]


def test_resolve_uses_virtual_substitution() -> None:
    assert resolve(Exists(x, (x > y) & (x < 1))) == (y < 1)
    assert resolve(ForAll(x, (x > y) | (x < z))) == (y < z)
    assert resolve(ForAll(x, Exists(y, y > x))) is true
    assert resolve(Exists(y, Eq(a*y + b, 0) & (y > 0))) == Or(And(Eq(a, 0), Eq(b, 0)), And(a > 0, b < 0), And(a < 0, b > 0))
    # a mixed prefix: the linear variable by substitution, the other by the CAD
    assert resolve(Exists(x, (x**2 < y) & Exists(z, (z > x) & (z < 1)))) == (y > 0)
