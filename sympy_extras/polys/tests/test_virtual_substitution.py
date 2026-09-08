from __future__ import annotations

from sympy import S, Eq, Ne, Or, And, Equivalent, true, false, Rational
from sympy.abc import x, y, z, a, b, c
from sympy.testing.pytest import raises

from sympy_extras.polys.virtual_substitution import (is_linear_in, eliminate_linear,
    linear_quantifier_elimination, is_quadratic_in, eliminate_quadratic, virtual_substitution_elimination)
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


def test_eliminate_quadratic() -> None:
    assert eliminate_quadratic(Eq(x**2 + b*x + c, 0), x) == (b**2 >= 4*c)
    assert eliminate_quadratic((x**2 < a) & (x > 1), x) == (a > 1)
    assert eliminate_quadratic(x**2 < 0, x) is false
    assert eliminate_quadratic(x**2 + 1 > 0, x) is true
    assert is_quadratic_in((x**2 + y*x > 0) & (x < 1), x) and not is_quadratic_in(x**3 > y, x)
    raises(ValueError, lambda: eliminate_quadratic(x**3 > y, x))
    # against the cylindrical algebraic decomposition
    formulas = [
        Eq(a*x**2 + b*x + c, 0), (x**2 + b*x + c > 0) & (x > 0), (x**2 < y) & (x > y - 1),
        (a*x**2 + 1 <= 0) | (x < a), Ne(x**2 - a, 0) & (x >= 0) & (x <= 1),
        (x**2 - 2*a*x + b > 0) & (x < a), Eq(x**2, a) & Eq(x, b), (x**2 >= a) & (x**2 <= b),
        (x**2 + a*x < 0) & (x**2 - b*x > 0), (a*x**2 + b*x + 1 >= 0) & (x < 0),
    ]
    import random
    rng = random.Random(5)
    for f in formulas:
        result = eliminate_quadratic(f, x)
        assert not result.has(x)
        parameters = sorted((s for s in f.free_symbols if s != x), key=lambda s: s.name)
        # the same truth value as the decomposition at random rational points
        for _ in range(8):
            point = {s: Rational(rng.randint(-4, 4), rng.choice([1, 2, 3])) for s in parameters}
            expected = resolve(Exists(x, f.subs(point)))
            actual = result.subs(point)
            assert actual in (true, false) and expected in (true, false), (f, point)
            assert actual == expected, (f, point, result)


def test_virtual_substitution_elimination() -> None:
    assert virtual_substitution_elimination(Eq(x**2 + b*x + c, 0), [('exists', x)]) == (b**2 >= 4*c, [])
    assert virtual_substitution_elimination(x**2 + b*x + c > 0, [('forall', x)]) == (b**2 < 4*c, [])
    assert virtual_substitution_elimination(x**3 > y, [('exists', x)]) == (x**3 > y, [('exists', x)])
    assert resolve(ForAll(x, x**2 + b*x + c > 0)) == (b**2 < 4*c)
    assert resolve(Exists(x, (x**2 < a) & (x > 1))) == (a > 1)
