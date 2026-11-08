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


def test_univariate_factor_over_gaussian_matches_sympy() -> None:
    rng = random.Random(5)
    for _ in range(60):
        e: s.Expr = s.Integer(1)
        for _ in range(rng.randint(1, 3)):
            p = sum(((rng.randint(-3, 3) + rng.randint(-3, 3) * s.I) * x**k for k in range(rng.randint(1, 2) + 1)),
                    s.Integer(0))
            if p != 0:
                e = e * p**rng.randint(1, 2)
        e = s.expand(e * s.Rational(rng.choice([1, 2, 3]), rng.choice([1, 2])))
        if not e.has(x):
            continue
        assert str(to_sympy(expr_from_sympy(e).factor(), evaluate=False)) == str(s.factor(e)), e


def test_cancel_with_floats_matches_sympy() -> None:
    rng = random.Random(7)
    atoms = [x, y, x*y, x**2, s.Float(0.5), s.Float(1.25), s.Float('0.1'), s.Rational(1, 3), s.Integer(2)]

    def rp() -> s.Expr:
        result: s.Expr = s.Integer(0)
        for _ in range(rng.randint(1, 3)):
            term: s.Expr = (s.Float(rng.choice([1, -2, 0.5, 3.75, -0.1])) if rng.random() < 0.5
                            else s.Integer(rng.randint(-3, 3)))
            for _ in range(rng.randint(0, 2)):
                term = term * rng.choice(atoms)
            result = result + term
        return result

    for _ in range(100):
        e = rp() / rp() + (rp() / rp() if rng.random() < 0.3 else s.Integer(0))
        if e.has(s.zoo, s.nan):
            continue
        assert str(to_sympy(expr_from_sympy(e).cancel(), evaluate=False)) == str(s.cancel(e)), e


def test_factor_with_floats_matches_sympy() -> None:
    rng = random.Random(1)
    for _ in range(50):
        e: s.Expr = s.Integer(1)
        for _ in range(rng.randint(1, 3)):
            p = sum((s.Float(rng.choice([1, -2, 0.5, 1.5, -0.25, 3, 0.1])) * rng.choice([x, y, x**2, s.Integer(1), x*y])
                     for _ in range(rng.randint(1, 3))), s.Integer(0))
            if p != 0:
                e = e * p**rng.randint(1, 2)
        e = s.expand(e)
        if not e.free_symbols:
            continue
        assert str(to_sympy(expr_from_sympy(e).factor(), evaluate=False)) == str(s.factor(e)), e


def test_cancel_with_gens_matches_sympy() -> None:
    z = s.Symbol('z')
    rng = random.Random(11)
    atoms = [x, y, z, x*y, y*z, x**2, s.Integer(2), s.Rational(1, 2), s.Integer(-3)]

    def rp() -> s.Expr:
        result: s.Expr = s.Integer(0)
        for _ in range(rng.randint(1, 3)):
            term: s.Expr = s.Integer(rng.randint(-3, 3))
            for _ in range(rng.randint(0, 2)):
                term = term * rng.choice(atoms)
            result = result + term
        return result

    for _ in range(100):
        c = rp()
        e = (rp() * c) / (rp() * c)
        if e.has(s.zoo, s.nan):
            continue
        gens = rng.choice([(x,), (y,), (x, y), (z, x)])
        ours = expr_from_sympy(e).cancel(*[expr_from_sympy(g) for g in gens])
        assert str(to_sympy(ours, evaluate=False)) == str(s.cancel(e, *gens)), (e, gens)
