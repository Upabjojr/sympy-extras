"""Oracle tests of simplify against SymPy."""
import random

import sympy

from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.simplify.simplify import simplify

s = sympy
x, y, z = s.symbols('x y z')
p = s.Symbol('p', positive=True)
n = s.Symbol('n', integer=True)


def _ours(e: sympy.Expr) -> str:
    return str(to_sympy(simplify(expr_from_sympy(e)), evaluate=False))


def test_simplify_matches_sympy() -> None:
    cases: list[sympy.Expr] = [
        s.sin(x)**2 + s.cos(x)**2, (x**3 + x**2 - x - 1) / (x**2 + 2 * x + 1), s.gamma(x + 1) / s.gamma(x),
        s.log(2) + s.log(3), (x + 1)**2 - x**2 - 2 * x, 1 / x + 1 / y, s.exp(x) * s.exp(y), s.sqrt(p**2),
        s.Abs(x) * s.Abs(y), s.sign(2 * x), s.log(p**2) / 2, s.cos(x)**4 - s.sin(x)**4, (s.exp(x) - s.exp(-x)) / 2,
        s.factorial(n + 1) / s.factorial(n), 1 / (s.sqrt(2) + 1), x / (x + y) + y / (x + y), s.tan(x) * s.cos(x),
        s.Abs(x + 1) / s.Abs(x**2 - 1), s.cosh(x)**2 - s.sinh(x)**2, 0.5 * x + x / 2,
        s.Piecewise((x + x, x > 0), (1, True)), s.binomial(n, 2), s.exp(2 * s.log(x)), s.log(x * y) - s.log(x),
        p / 4 + s.log(x**2 / p) + s.gamma(s.pi * y) + s.pi / 4 + 2]
    for e in cases:
        assert _ours(e) == str(s.simplify(e)), e


def test_simplify_random() -> None:
    rng = random.Random(3)
    atoms = [x, y, p, s.Integer(2), s.Rational(1, 2), s.pi]
    funcs = [s.sin, s.cos, s.exp, s.log, s.sqrt, lambda a: a**2, lambda a: 1 / a, s.tan, s.gamma, s.Abs]

    def rnd(d: int) -> sympy.Expr:
        if d == 0 or rng.random() < .25:
            return rng.choice(atoms)
        r = rng.random()
        if r < .35:
            return rnd(d - 1) + rnd(d - 1)
        if r < .7:
            return rnd(d - 1) * rnd(d - 1)
        return rng.choice(funcs)(rnd(d - 1))
    for _ in range(25):
        e = rnd(3)
        assert _ours(e) == str(s.simplify(e)), e


def test_inversecombine_and_kroneckersimp_match_sympy() -> None:
    from sympy.simplify.simplify import inversecombine as sympy_inversecombine, kroneckersimp as sympy_kroneckersimp
    from sympy_extras.simplify.simplify import inversecombine, kroneckersimp
    i, j = s.symbols('i j')
    K = s.KroneckerDelta
    for e in [s.log(s.exp(x)) + s.exp(s.log(y)), s.atan(s.tan(x)), s.asinh(s.sinh(s.exp(s.log(x)))),
              s.erf(s.erfinv(x)) * s.tan(s.atan(y)), s.sin(s.asin(x)) + s.acosh(s.cosh(x))]:
        assert str(to_sympy(inversecombine(expr_from_sympy(e)), evaluate=False)) == str(sympy_inversecombine(e)), e
    for e in [1 + K(0, i) * K(1, i), K(i, j) * K(0, i) * K(1, j), x * K(i, 0) * K(i, 0) + K(i, 2) * K(i, 1)]:
        assert str(to_sympy(kroneckersimp(expr_from_sympy(e)), evaluate=False)) == str(sympy_kroneckersimp(e)), e


def test_simplify_keeps_dummies_through_sympy_steps() -> None:
    # hyper_as_trig's dummy must survive the factor fallback (domain ZZ_I)
    e = x + s.sqrt(1 - s.sinh(s.sinh(x))**2) / s.sinh(s.sinh(x))
    assert _ours(e) == str(s.simplify(e))
