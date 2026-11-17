"""The typed ``Poly`` over the fraction fields ``ZZ(a, b)`` and ``QQ(a)``
against SymPy (the oracle): the domains ``construct_domain`` chooses, and
the operations on random polynomials whose coefficients are rational
functions of the parameters, compared through the printed forms (SymPy
prints every coefficient as its element of the field in lowest terms)."""
from __future__ import annotations

import random
from typing import Callable, Union

import pytest
import sympy
from sympy.polys.polyerrors import BasePolynomialError

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy, to_sympy
from sympy_extras.polys.polyerrors import CoercionFailed, ExactQuotientFailed, OptionError
from sympy_extras.polys.polytools import Poly

sx, sy, sz, sa, sb = sympy.symbols('x y z a b')
x, y, z, a, b = (Symbol(s) for s in 'xyzab')

Result = Union[Poly, Expr, int, bool, tuple[Expr, list[tuple[Poly, int]]], tuple[Poly, Poly],
               tuple[Expr, Poly], tuple[Expr, Poly, Poly], list[Expr]]


def printed(value: Result) -> str:
    """The printed form of a result as SymPy prints its own."""
    if isinstance(value, Expr):
        return str(to_sympy(value, evaluate=False))
    if isinstance(value, list):
        return str([to_sympy(v, evaluate=False) for v in value if isinstance(v, Expr)])
    if isinstance(value, tuple) and value and isinstance(value[0], Expr):
        head = to_sympy(value[0], evaluate=False)
        rest = ', '.join(str(v) for v in value[1:])
        return '(%s, %s)' % (head, rest)
    if isinstance(value, tuple):
        return '(%s)' % ', '.join(str(v) for v in value)
    return str(value)


def theirs_printed(value: object) -> str:
    if isinstance(value, tuple) and value and isinstance(value[0], sympy.Expr):
        return '(%s, %s)' % (value[0], ', '.join(str(v) for v in value[1:]))
    if isinstance(value, tuple):
        return '(%s)' % ', '.join(str(v) for v in value)
    return str(value)


def random_rational(rng: random.Random, params: list[sympy.Symbol]) -> sympy.Expr:
    def poly() -> sympy.Expr:
        return sympy.Add(*[rng.randint(-3, 3) * sympy.Mul(*[p**rng.randint(0, 2) for p in params])
                           for _ in range(rng.randint(1, 3))])
    numer, denom = poly(), poly()
    if denom == 0:
        denom = sympy.Integer(1)
    return numer / denom * rng.choice([1, 1, sympy.Rational(1, 2)])


def random_fraction_poly(rng: random.Random, gens: list[sympy.Symbol], params: list[sympy.Symbol],
                         deg: int) -> sympy.Expr:
    return sympy.Add(*[random_rational(rng, params) * sympy.Mul(*[g**rng.randint(0, deg) for g in gens])
                       for _ in range(rng.randint(1, 3))])


@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_random_fraction_polynomials_match_sympy(seed: int) -> None:
    rng = random.Random(seed)
    different: list[tuple[str, str, str, str, str]] = []
    count = 0
    for _ in range(30):
        params = rng.choice([[sy], [sy], [sa, sb]])
        gens = rng.choice([[sx], [sx], [sx, sz]])
        e = random_fraction_poly(rng, gens, params, 2)
        f = random_fraction_poly(rng, gens, params, 1) * rng.choice([1, random_fraction_poly(rng, gens, params, 1)])
        try:
            P = sympy.Poly(e, *gens)
            Q = sympy.Poly(f, *gens)
            P, Q = P.unify(Q)
        except BasePolynomialError:
            continue
        if not P.domain.is_FractionField or Q.is_zero or P.is_zero:
            continue
        count += 1
        # both over SymPy's unified domain (by its name)
        p = Poly(expr_from_sympy(e), *[expr_from_sympy(g) for g in gens], domain=str(P.domain))
        q = Poly(expr_from_sympy(f), *p.gens, domain=str(P.domain))
        point = sympy.Rational(rng.randint(-5, 5), rng.choice([1, 2, 3]))
        checks: list[tuple[str, Callable[[], Result], Callable[[], object]]] = [
            ('repr', lambda: p, lambda: P), ('degree', lambda: p.degree(), lambda: P.degree()),
            ('LC', lambda: p.LC(), lambda: P.LC()), ('coeffs', lambda: p.coeffs(), lambda: P.coeffs()),
            ('add', lambda: p + q, lambda: P + Q), ('sub', lambda: p - q, lambda: P - Q),
            ('mul', lambda: p * q, lambda: P * Q), ('pow', lambda: p**2, lambda: P**2),
            ('neg', lambda: -p, lambda: -P), ('diff', lambda: p.diff(), lambda: P.diff(P.gens[0])),
            ('gcd', lambda: p.gcd(q), lambda: P.gcd(Q)), ('lcm', lambda: p.lcm(q), lambda: P.lcm(Q)),
            ('monic', lambda: p.monic(), lambda: P.monic()), ('content', lambda: p.content(), lambda: P.content()),
            ('primitive', lambda: p.primitive(), lambda: P.primitive()),
            ('cancel', lambda: p.cancel(q), lambda: P.cancel(Q)),
            ('factor_list', lambda: p.factor_list(), lambda: P.factor_list()),
            ('sqf_list', lambda: p.sqf_list(), lambda: P.sqf_list()),
            ('eval', lambda: p.eval(p.gens[0], expr_from_sympy(point)), lambda: P.eval(P.gens[0], point)),
            ('equal', lambda: p == Poly(p.as_expr(), *p.gens), lambda: True),
            ('exquo', lambda: (p * q).exquo(q), lambda: (P * Q).exquo(Q))]
        if len(gens) == 1:
            checks += [('div', lambda: p.div(q), lambda: P.div(Q)), ('quo', lambda: p.quo(q), lambda: P.quo(Q)),
                       ('rem', lambda: p.rem(q), lambda: P.rem(Q)),
                       ('resultant', lambda: p.resultant(q), lambda: P.resultant(Q)),
                       ('discriminant', lambda: p.discriminant(), lambda: P.discriminant())]
        for name, ours, theirs in checks:
            try:
                expected = theirs()
            except (ZeroDivisionError, BasePolynomialError):
                continue
            if name == 'degree' and expected == -sympy.oo:
                expected = -1
            if isinstance(expected, list):
                expected_text = str(expected)
            else:
                expected_text = theirs_printed(expected)
            got = printed(ours())
            if got != expected_text:
                different.append((name, str(e), str(f), got, expected_text))
    assert count >= 10
    assert not different, (sorted(set((d[0], d[4][-14:]) for d in different)), different[:3])


def test_construct_domain_choices_match_sympy() -> None:
    """The domain of the coefficients, as SymPy's ``construct_domain``
    chooses it."""
    cases: list[tuple[sympy.Expr, tuple[sympy.Symbol, ...]]] = [
        (sx / sy, (sx,)), (sx / (2 * sy) + sympy.Rational(1, 3), (sx,)), (sx / sy + sx / sy**2, (sx,)),
        (sx * sa / sb + sb, (sx,)), ((sx + 1) / (sy * sz), (sx,)), (sx / 2 + sy, (sx,)), (sx * sy + 1, (sx,)),
        (sx**2 / 3, (sx,)), (sx / (sy + 1) + 1 / (2 * sy), (sx,)), (sx * sy / sz + 1, (sx, sy)),
        (sx / sy + sz, (sx, sz)), (sx * sympy.Rational(1, 2) / (sy + sympy.Rational(1, 3)), (sx,)),
        (sx / (sy**2 - 1) - sx / (sy - 1), (sx,)), (sx + 1 / sy - 1 / sy, (sx,)),
        (sympy.Symbol('_x') + sympy.Symbol('_z') / 2 - sympy.Symbol('_z') / (2 * sy), (sympy.Symbol('_x'),)),
        (sx**2 * sa / (sa + sb) - sb * sx / sa, (sx,)), (sx / (2 * sy), (sx,))]
    for e, gens in cases:
        theirs = sympy.Poly(e, *gens)
        ours = Poly(expr_from_sympy(e), *[expr_from_sympy(g) for g in gens])
        assert (ours.domain, str(ours)) == (str(theirs.domain), str(theirs)), e


def test_poly_over_a_named_fraction_field() -> None:
    p = Poly(x**2 - y**2, x, domain='ZZ(y)')
    assert str(p) == "Poly(x**2 - y**2, x, domain='ZZ(y)')" and p.is_fraction_field
    assert str(p.gcd(Poly(y * x + y**2, x, domain='ZZ(y)'))) == "Poly(x + y, x, domain='ZZ(y)')"
    assert str(Poly(x / 2, x, domain='ZZ(y)')) == "Poly(1/2*x, x, domain='ZZ(y)')"
    assert str(Poly(x * y / 3, x, domain='QQ(y)')) == "Poly(y/3*x, x, domain='QQ(y)')"
    assert str(Poly(x * (y / 2) / (y + Rational(1, 3)), x, domain='QQ(y)')) == \
        str(sympy.Poly(sx * (sy / 2) / (sy + sympy.Rational(1, 3)), sx, domain='QQ(y)'))
    assert str(Poly(x + 1, x, domain='ZZ[y]')) == "Poly(x + 1, x, domain='ZZ[y]')"
    assert str(Poly(Poly(x + y, x), domain='QQ(y,z)')) == "Poly(x + y, x, domain='QQ(y,z)')"
    with pytest.raises(CoercionFailed):
        Poly(x / y, x, domain='ZZ[y]')
    with pytest.raises(CoercionFailed):
        Poly(x / 2, x, domain='ZZ')
    with pytest.raises(OptionError):
        Poly(x, x, domain='RR')
    # the unification of SymPy's unify_composite
    assert (Poly(x / 2 + y, x) + Poly(x / y, x)).domain == 'ZZ(y)'
    assert (Poly(x / 2 + y, x, domain='QQ(y)') + Poly(x, x, domain='QQ[y]')).domain == 'QQ(y)'
    assert (Poly(x / y, x) + Poly(x / 2, x)).domain == 'QQ(y)'
    assert Poly(x / y, x) == Poly(2 * x / (2 * y), x) and Poly(x / y, x) != Poly(x, x, domain='ZZ(y)')
    assert Poly(y * x, x, domain='ZZ(y)') == Poly(y * x, x) and Poly(y * x, x).is_fraction_field is False


def test_polynomial_domains_go_to_the_fraction_field() -> None:
    """SymPy's ``div(auto=True)`` and ``monic`` over ``ZZ[y]`` work over
    ``ZZ(y)`` and come back to ``ZZ[y]`` when they can."""
    assert str(Poly(x, x, domain='ZZ[y]').div(Poly(y * x, x, domain='ZZ[y]'))) == \
        "(Poly(1/y, x, domain='ZZ(y)'), Poly(0, x, domain='ZZ(y)'))"
    assert str(Poly(x**2 * y + 1, x).div(Poly(y * x + 1, x))) == \
        "(Poly(x - 1/y, x, domain='ZZ(y)'), Poly((y + 1)/y, x, domain='ZZ(y)'))"
    assert str(Poly(x**2 * y - y, x).div(Poly(y * x - y, x))) == \
        str(sympy.Poly(sx**2 * sy - sy, sx).div(sympy.Poly(sy * sx - sy, sx)))
    assert str(Poly(y * x + 1, x).monic()) == "Poly(x + 1/y, x, domain='ZZ(y)')"
    # a number as the leading coefficient: QQ[y] (SymPy goes to ZZ(y) all
    # the same; kept as it was before the fraction fields)
    assert str(Poly(2 * x + y, x).monic()) == "Poly(x + y/2, x, domain='QQ[y]')"


def test_fraction_field_operations_by_hand() -> None:
    f = Poly(x**2 / y - 1 / (y + 1), x)
    g = Poly(x / y + 1, x)
    assert str(f) == "Poly(1/y*x**2 - 1/(y + 1), x, domain='ZZ(y)')"
    q, r = f.div(g)
    assert q * g + r == f
    assert str(q) == "Poly(x - y, x, domain='ZZ(y)')" and str(r) == "Poly((y**2 + y - 1)/(y + 1), x, domain='ZZ(y)')"
    assert printed(f.eval(x, Integer(2))) == '(3*y + 4)/(y**2 + y)'
    assert str(f.eval(x, y)) == str(sympy.Poly(sx**2 / sy - 1 / (sy + 1), sx).eval(sx, sy))
    assert printed(f.resultant(g)) == '(y**2 + y - 1)/(y**3 + y**2)'
    assert printed(f.discriminant()) == '4/(y**2 + y)'
    assert printed(f.content()) == '1/(y**2 + y)'
    c, p, s = f.cancel(g)
    assert printed(c) == '1/(y + 1)' and str(p) == "Poly((y + 1)*x**2 - y, x, domain='ZZ(y)')"
    assert str(s) == "Poly(x + y, x, domain='ZZ(y)')"
    assert f.diff(x) == Poly(2 * x / y, x) and f**0 == Poly(Integer(1), x, domain='ZZ(y)')
    assert (f - f).is_zero and (f * g).exquo(g) == f
    with pytest.raises(ExactQuotientFailed):
        f.exquo(g)
    h = Poly(x * y / z + 1, x, y)
    assert str(h) == "Poly(1/z*x*y + 1, x, y, domain='ZZ(z)')"
    assert str(h.resultant(Poly(x - z, x, y))) == str(sympy.Poly(sx * sy / sz + 1, sx, sy).resultant(
        sympy.Poly(sx - sz, sx, sy)))
    assert printed(Poly((x + 1 / y)**2 * (x - 1), x).sqf_list()) == \
        "(1, [(Poly(x - 1, x, domain='ZZ(y)'), 1), (Poly(x + 1/y, x, domain='ZZ(y)'), 2)])"
