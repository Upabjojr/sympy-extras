"""Oracle tests of gammasimp and combsimp against SymPy."""
import random

import sympy

from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.simplify.gammasimp import combsimp, gammasimp

s = sympy
x, y = s.symbols('x y')
n, k = s.symbols('n k', integer=True)


def _ours(f: object, e: sympy.Expr) -> str:
    assert f is gammasimp or f is combsimp
    result = gammasimp(expr_from_sympy(e)) if f is gammasimp else combsimp(expr_from_sympy(e))
    return str(to_sympy(result, evaluate=False))


def test_gammasimp_identities() -> None:
    cases: list[sympy.Expr] = [
        s.gamma(x) / s.gamma(x - 3), s.gamma(x) * s.gamma(1 - x), s.gamma(x) * s.gamma(x + s.S.Half) / s.gamma(2 * x),
        s.gamma(x) * s.gamma(x + s.Rational(1, 3)) * s.gamma(x + s.Rational(2, 3)), s.gamma(s.Rational(7, 3)),
        x * s.gamma(x) / s.gamma(x + 1), s.beta(x, y), s.rf(x, 3) * s.gamma(x), s.gamma(x) * s.gamma(-x),
        s.exp(s.gamma(x + 1) / s.gamma(x)), (s.gamma(x + 2) - s.gamma(x)) / s.gamma(x)]
    for e in cases:
        assert _ours(gammasimp, e) == str(s.gammasimp(e)), e


def test_combsimp_identities() -> None:
    cases: list[sympy.Expr] = [
        s.factorial(n) / s.factorial(n - 3), s.binomial(n + 1, k + 1) / s.binomial(n, k),
        s.factorial(n + k) / (s.factorial(n) * s.factorial(k)), s.rf(n, 3) / s.factorial(n + 2),
        s.ff(n, k) / s.factorial(n), s.factorial(2 * n) / s.factorial(n)**2, s.binomial(n, k) / s.binomial(n, k - 1),
        s.gamma(x + 1) / s.gamma(x)]
    for e in cases:
        assert _ours(combsimp, e) == str(s.combsimp(e)), e


def test_gammasimp_random_products() -> None:
    rng = random.Random(1)
    shifts = [-2, -1, 0, 1, 2, s.S.Half, -s.S.Half, s.Rational(1, 3)]
    for _ in range(15):
        e: sympy.Expr = s.S.One
        for _ in range(rng.randint(2, 4)):
            e *= s.gamma(rng.choice([x, y, 2 * x, x + y]) + rng.choice(shifts))**rng.choice([1, -1, 2])
        assert _ours(gammasimp, e) == str(s.gammasimp(e)), e
