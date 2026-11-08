"""Oracle test of ratsimp against SymPy."""
import random

import sympy

from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.simplify.ratsimp import ratsimp

s = sympy
x, y, z = s.symbols('x y z')


def test_ratsimp_matches_sympy() -> None:
    rng = random.Random(1)
    atoms = [x, y, z, s.sqrt(2), s.pi, s.I]

    def poly(d: int) -> sympy.Expr:
        return s.Add(*[rng.choice([1, 2, -1, s.Rational(1, 2), 3])
                       * s.Mul(*[rng.choice(atoms)**rng.randint(0, 2) for _ in range(rng.randint(0, d))])
                       for _ in range(rng.randint(1, 3))])
    for _ in range(60):
        e = poly(3) / poly(2) + rng.choice([0, 1 / x, poly(2) / poly(1), s.Rational(1, 2)])
        assert str(to_sympy(ratsimp(expr_from_sympy(e)), evaluate=False)) == str(s.ratsimp(e)), e
