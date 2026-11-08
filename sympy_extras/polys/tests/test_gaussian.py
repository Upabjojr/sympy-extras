"""The Gaussian domains and the cancellation over them, against SymPy."""
import random

import sympy
from sympy.polys.domains import QQ_I as SQQ_I, ZZ_I as SZZ_I

from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.polys.domains.gaussiandomains import GaussianInteger, QQ_I, ZZ_I

s = sympy
x, y = s.symbols('x y')


def test_gaussian_integer_arithmetic_matches_sympy() -> None:
    rng = random.Random(1)
    for _ in range(300):
        a = (rng.randint(-20, 20), rng.randint(-20, 20))
        b = (rng.randint(-20, 20), rng.randint(-20, 20))
        if b == (0, 0):
            continue
        A, B = GaussianInteger(*a), GaussianInteger(*b)
        SA, SB = SZZ_I(*a), SZZ_I(*b)
        q, r = ZZ_I.divmod(A, B)
        sq, sr = divmod(SA, SB)
        assert (q.x, q.y, r.x, r.y) == (sq.x, sq.y, sr.x, sr.y)
        g = ZZ_I.gcd(A, B)
        sg = SZZ_I.gcd(SA, SB)
        assert (g.x, g.y) == (sg.x, sg.y)
        if A:
            u = ZZ_I.canonical_unit(A)
            su = SZZ_I.canonical_unit(SA)
            assert (u.x, u.y) == (su.x, su.y)
        assert str(QQ_I.to_expr(QQ_I.invert(QQ_I.convert(B, ZZ_I)))) == str(SQQ_I.to_sympy(1 / SQQ_I.convert(SB)))


def test_cancel_with_gaussian_coefficients_matches_sympy() -> None:
    rng = random.Random(2)
    atoms = [x, y, s.I, s.I * x, x - s.I, s.Rational(1, 2), s.Integer(3), y + s.I]

    def rp() -> s.Expr:
        result: s.Expr = s.Integer(0)
        for _ in range(rng.randint(1, 3)):
            term: s.Expr = s.Integer(rng.randint(-3, 3))
            for _ in range(rng.randint(0, 2)):
                term = term * rng.choice(atoms)
            result = result + term
        return result

    for _ in range(120):
        e = rp() / rp() + rp() / rp()
        if e.has(s.zoo, s.nan):
            continue
        assert str(to_sympy(expr_from_sympy(e).cancel(), evaluate=False)) == str(s.cancel(e)), e
    for e in [(x**2 + 1)/(x + s.I), (x + s.I)**2/(x**2 - 1 + 2*s.I*x), (2*s.I*x + 2)/(4*x - 4*s.I),
              s.sqrt(s.I)*s.I*x/(x + x**2)]:
        assert str(to_sympy(expr_from_sympy(e).cancel(), evaluate=False)) == str(s.cancel(e)), e
