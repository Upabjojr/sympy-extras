"""The typed ``Poly`` against SymPy (the oracle): random polynomials over
``ZZ``, ``QQ``, ``ZZ[a, b]``, ``QQ[a, b]`` with and without explicit
generators, compared through the printed forms. The resultant is checked
against the Sylvester determinant: SymPy returns ``Res(g, f)`` when
``deg f < deg g`` (a sign error when both degrees are odd)."""
from __future__ import annotations

import random
from fractions import Fraction
from typing import Callable, Union

import pytest
import sympy
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.subresultants_qq_zz import sylvester

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import Symbol
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy
from sympy_extras.functions import sin, sqrt
from sympy_extras.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy_extras.polys.polytools import Poly, degree, factor_list, gcd, lcm, resultant

sx, sy, sz, sa, sb = sympy.symbols('x y z a b')
Result = Union[Poly, Expr, int, tuple[Expr, list[tuple[Poly, int]]], tuple[Expr, list[tuple[Expr, int]]],
               tuple[Expr, list[tuple[Expr, Expr]]], tuple[Poly, Poly], list[Expr],
               list[tuple[int, ...]]]


def printed(value: Result) -> str:
    if isinstance(value, Expr):
        return str(to_sympy(value, evaluate=False))
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], Expr) and isinstance(value[1], list):
        return str((to_sympy(value[0]), value[1]))
    if isinstance(value, list) and value and isinstance(value[0], Expr):
        return str([sympy_expr(to_sympy(v)) for v in value if isinstance(v, Expr)])
    return str(value)


def random_polynomial(rng: random.Random, variables: list[sympy.Symbol], deg: int) -> sympy.Expr:
    return sympy.Add(*[rng.randint(-3, 3) * sympy.Mul(*[v**rng.randint(0, deg) for v in variables])
                       for _ in range(rng.randint(1, 4))])


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_random_polynomials_match_sympy(seed: int) -> None:
    rng = random.Random(seed)
    different: list[tuple[str, str, str, str, str]] = []
    for _ in range(60):
        variables = rng.choice([[sx], [sx, sy], [sx, sa], [sx, sy, sa, sb], [sx, sa, sb], [sx, sy, sz, sa]])
        gens = rng.choice([(), (sx,), tuple(variables), (sx, variables[-1])])
        e = sympy.expand(random_polynomial(rng, variables, 3) * rng.choice([1, sympy.Rational(1, 2), sympy.Rational(-3, 4)]))
        f = sympy.expand(rng.choice([random_polynomial(rng, variables, 2), random_polynomial(rng, variables, 1) * e,
                                     sympy.Integer(0)]))
        try:
            P = sympy.Poly(e, *gens)
            Q = sympy.Poly(f, *P.gens)
        except BasePolynomialError:
            continue
        p = Poly(expr_from_sympy(e), *[expr_from_sympy(g) for g in gens])
        q = Poly(expr_from_sympy(f), *p.gens)
        checks: list[tuple[str, Callable[[], Result], Callable[[], object]]] = [
            ('repr', lambda: p, lambda: P), ('degree', lambda: p.degree(), lambda: P.degree()),
            ('LC', lambda: p.LC(), lambda: P.LC()), ('coeffs', lambda: p.coeffs(), lambda: P.coeffs()),
            ('monoms', lambda: p.monoms(), lambda: P.monoms()), ('add', lambda: p + q, lambda: P + Q),
            ('mul', lambda: p * q, lambda: P * Q), ('sub', lambda: p - q, lambda: P - Q),
            ('diff', lambda: p.diff(), lambda: P.diff(P.gens[0])), ('gcd', lambda: p.gcd(q), lambda: P.gcd(Q)),
            ('lcm', lambda: p.lcm(q), lambda: P.lcm(Q))]
        if len(P.gens) == 1 and P.domain in (sympy.ZZ, sympy.QQ) and P.degree() > 0:
            checks += [('factor_list', lambda: p.factor_list(), lambda: P.factor_list()),
                       ('sqf_list', lambda: p.sqf_list(), lambda: P.sqf_list()),
                       ('discriminant', lambda: p.discriminant(), lambda: P.discriminant())]
            if Q.domain in (sympy.ZZ, sympy.QQ) and not Q.is_zero:
                checks.append(('div', lambda: p.div(q), lambda: P.div(Q)))
        for name, ours, theirs in checks:
            try:
                expected = theirs()
            except ZeroDivisionError:
                continue
            if name == 'degree' and expected == -sympy.oo:
                expected = -1
            if printed(ours()) != str(expected):
                different.append((name, str(e), str(f), printed(ours()), str(expected)))
        if not P.is_zero and not Q.is_zero:
            r = p.resultant(q)
            ours_r = sympy_expr(to_sympy(r.as_expr() if isinstance(r, Poly) else r))
            g = P.gens[0]
            if P.degree() > 0 or Q.degree() > 0:
                if P.degree() > 0 and Q.degree() > 0:
                    exact = sylvester(e, f, g).det()
                else:
                    value = P.resultant(Q)
                    exact = value.as_expr() if isinstance(value, sympy.Poly) else value
                if sympy.expand(ours_r - sympy_expr(exact)) != 0:
                    different.append(('resultant', str(e), str(f), str(ours_r), str(exact)))
    assert not different, different[:5]


def test_construction_printing_errors() -> None:
    x, y, z, a, b = (Symbol(s) for s in 'xyzab')
    assert str(Poly(y*x + sin(z) + sqrt(2))) == str(sympy.Poly(sy*sx + sympy.sin(sz) + sympy.sqrt(2)))
    assert str(Poly(sqrt(x) + x)) == str(sympy.Poly(sympy.sqrt(sx) + sx))
    assert str(Poly(a*x + b*x, x)) == "Poly((a + b)*x, x, domain='ZZ[a,b]')"
    assert str(Poly(a/2*x, x)) == "Poly(a/2*x, x, domain='QQ[a]')"
    assert str(Poly(-x**2 + 3, x)) == str(sympy.Poly(-sx**2 + 3, sx))
    with pytest.raises(PolynomialError):
        Poly(1/x, x)
    with pytest.raises(PolynomialError):
        Poly(sin(x), x)
    with pytest.raises(GeneratorsNeeded):
        Poly(sqrt(2) - sqrt(2))
    assert degree(x**3*y + y, y) == 1
    assert printed(gcd(x**2 - 1, x**2 + 2*x + 1)) == 'x + 1'
    assert printed(lcm(x**2 - 1, x + 1)) == 'x**2 - 1'
    assert printed(resultant(x**2 + a, x - b, x)) == str(sympy.resultant(sx**2 + sa, sx - sb, sx))
    assert printed(factor_list(2*x**3 - 2*x, x)) == str(sympy.factor_list(2*sx**3 - 2*sx, sx))


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_cancel_matches_sympy(seed: int) -> None:
    from sympy_extras.polys.polytools import cancel
    rng = random.Random(seed)
    atoms: list[sympy.Expr] = [sx, sy, sa, sympy.sin(sx), sympy.exp(sx), 2**sx, sympy.sqrt(2), sympy.sqrt(sx),
                               sx**sympy.Rational(3, 2)]

    def rational(k: int) -> sympy.Expr:
        return sympy.Add(*[rng.randint(-3, 3) * sympy.Rational(rng.choice([1, 2]), rng.choice([1, 3]))
                           * sympy.Mul(*[rng.choice(atoms[:k])**rng.randint(0, 2) for _ in range(2)])
                           for _ in range(rng.randint(1, 3))])

    different: list[tuple[str, str, str]] = []
    for _ in range(80):
        k = rng.choice([3, 3, len(atoms)])
        c = rational(k)
        shape = rng.randint(0, 2)
        if shape == 0:
            e = rational(k) * c / (rational(k) * c)
        elif shape == 1:
            e = rational(k) / rational(k) + rational(k) / rational(k)
        else:
            e = rational(k) * c / c + 1 / rational(k)
        if e.has(sympy.zoo, sympy.nan):
            continue
        theirs = sympy.cancel(e)
        ours = str(to_sympy(cancel(expr_from_sympy(e)), evaluate=False))
        if ours != str(theirs):
            different.append((str(e), ours, str(theirs)))
    assert not different, different[:3]


def test_invert_matches_sympy() -> None:
    import random
    import sympy
    from sympy_extras.interop import expr_from_sympy, to_sympy
    from sympy_extras.polys.polyerrors import NotInvertible
    from sympy_extras.polys.polytools import invert
    rng = random.Random(3)
    sx = sympy.Symbol('x')

    def poly(d: int) -> sympy.Expr:
        return sympy.Add(*[rng.choice([0, 1, -1, 2, sympy.Rational(1, 2), 3]) * sx**i for i in range(rng.randint(1, d) + 1)])
    for _ in range(80):
        f, g = poly(4), poly(3)
        try:
            theirs = str(sympy.invert(f, g))
        except sympy.polys.polyerrors.NotInvertible:
            theirs = 'not invertible'
        try:
            ours = str(to_sympy(invert(expr_from_sympy(f), expr_from_sympy(g)), evaluate=False))
        except NotInvertible:
            ours = 'not invertible'
        assert ours == theirs, (f, g)


def test_invert_of_numbers_and_several_variables_matches_sympy() -> None:
    # computed natively since the SymPy calls moved to polytools_sympy:
    # several generators are a ValueError (SymPy's DMP.invert), two
    # rationals not both integers are never invertible (SymPy's QQ.gcdex)
    from sympy_extras.polys.polyerrors import NotInvertible
    from sympy_extras.polys.polytools import invert
    sx, sy = sympy.symbols('x y')
    half, third = sympy.Rational(1, 2), sympy.Rational(1, 3)
    pairs = [(sx*sy + 1, sy**2 + 1), (sx + 1, sx**2 + sy), (sympy.sqrt(2), sympy.sqrt(3)), (half, 3), (half, third),
             (sympy.Integer(3), sympy.Rational(7, 2)), (sympy.Rational(2, 3), 5), (0, sympy.Rational(7, 2)),
             (sympy.Rational(5, 3), sympy.Rational(2, 3)), (6, 4), (2, -7), (0, 1), (sympy.I, 3), (2 + sympy.I, 3)]
    for f, g in pairs:
        try:
            theirs = str(sympy.invert(f, g))
        except sympy.polys.polyerrors.NotInvertible:
            theirs = 'not invertible'
        except ValueError as e:
            theirs = 'ValueError %s' % e
        try:
            ours = str(to_sympy(invert(expr_from_sympy(sympy.sympify(f)), expr_from_sympy(sympy.sympify(g)))))
        except NotInvertible:
            ours = 'not invertible'
        except ValueError as e:
            ours = 'ValueError %s' % e
        assert ours == theirs, (f, g)


def test_gcd_list_and_lcm_list_match_sympy() -> None:
    import sympy
    from sympy_extras.interop import expr_from_sympy, to_sympy
    from sympy_extras.polys.polytools import gcd_list, lcm_list
    x, n, a = sympy.symbols('x n a')
    R = sympy.Rational
    cases = [[2*n + 2, 4*n**2 - 4], [n**2 - 1, n + 1, 3*n + 3], [sympy.Integer(6), R(4, 3)], [2*x*n, 6*n**2],
             [-n - 1, 2*n + 2], [a*n + a, n**2 + n], [R(1, 2)*n + R(1, 2), n**2 - 1], [-2*n**2 + 2, 4*n - 4],
             [R(3, 2)*n, R(9, 4)*n**2], [x**2*n - n, x*n + n], [n + 1]]
    for c in cases:
        native = [expr_from_sympy(e) for e in c]
        assert sympy.srepr(to_sympy(gcd_list(native))) == sympy.srepr(sympy.gcd_list(c)), c
        assert sympy.srepr(to_sympy(lcm_list(native))) == sympy.srepr(sympy.lcm_list(c)), c
    assert str(gcd_list([])) == '0' and str(lcm_list([])) == '1'


def test_discriminant_with_a_polynomial_leading_coefficient() -> None:
    # Res(f, f')/lc(f) was divided as an expression: with the leading
    # coefficient x + 1 in y the quotient 1/(x + 1) was left in it and
    # Poly(...) raised "1/(x + 1) contains an element of the set of
    # generators"; the division is now exact on the representation
    x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    f = Poly((x + 1)*y**2 + y + x, y, x)
    d = f.discriminant()
    assert isinstance(d, Poly) and d.gens == (x,)
    assert sympy.expand(to_sympy(d.as_expr())) == sympy.discriminant((sx + 1)*sy**2 + sy + sx, sy)
    g = Poly((x*y - 1)*(x + y)*y**2 + x**2, y, x)
    expected = sympy.discriminant(sympy.expand((sx*sy - 1)*(sx + sy)*sy**2 + sx**2), sy)
    found = g.discriminant()
    assert isinstance(found, Poly) and sympy.expand(to_sympy(found.as_expr()) - expected) == 0
    # with a parameter, and univariate
    h = Poly(a*x**2 + x + 1, x).discriminant()
    assert isinstance(h, Expr) and sympy.expand(to_sympy(h) - (1 - 4*sa)) == 0
    assert str(Poly(2*x**3 - x, x).discriminant()) == str(sympy.discriminant(2*sx**3 - sx, sx)) == '8'


def test_equality_and_hash_on_the_representation() -> None:
    # equal polynomials are equal and hash alike, over ZZ and QQ, and over
    # polynomial domains with the same parameters in another order
    x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')
    p, q = Poly(2*x + 2, x), Poly(x + 1, x)*2
    assert p == q and hash(p) == hash(q)
    assert Poly(x*y + 1, x, y) != Poly(x*y + 1, y, x)
    r = Poly(a*x + b*x, x)
    s = Poly(b*x, x) + Poly(a*x, x)
    assert r == s and hash(r) == hash(s)
    assert Poly(x/2, x) == Poly.from_dict({(1,): Fraction(1, 2)}, x)
    assert Poly.from_dict({(1, 0): 3, (0, 2): -1}, x, y) == Poly(3*x - y**2, x, y)
    assert Poly.from_dict({(1, 0): 3, (0, 2): -1}, x, y).domain == 'ZZ'
