"""``Poly`` of the Rust port against ``sympy_extras.polys.polytools.Poly``:
construction with and without generators (the domains ``ZZ``, ``QQ``,
``ZZ[a, b]``, ``QQ[a, b]`` and the errors), printing, terms, degrees and
the arithmetic, gcd, lcm, cofactors, cancel, division and exact quotient
of random polynomials sharing a factor, compared through ``str``."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed


def failed(f: Callable[[], Any]) -> Any:
    try:
        return f()
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def python_report(e: Any, other: Any, gens: list[Any]) -> dict[str, Any]:
    from sympy_extras.polys.polytools import Poly
    out: dict[str, Any] = {}
    try:
        f = Poly(e, *gens)
    except Exception as exc:
        out['poly'] = 'ERROR ' + type(exc).__name__
        return out
    out['poly'] = str(f)
    out['gens'] = [str(g) for g in f.gens]
    out['as_expr'] = str(f.as_expr())
    out['degrees'] = [f.degree(g) for g in f.gens]
    out['total_degree'] = f.total_degree()
    out['terms'] = repr([(list(m), str(c)) for m, c in f.terms()]).replace("'", '"')
    out['pow'] = str(f**2)
    out['diff'] = str(f.diff(f.gens[0]))
    try:
        g = Poly(other, *f.gens)
    except Exception as exc:
        out['other'] = 'ERROR ' + type(exc).__name__
        return out
    out['other'] = str(g)
    out['add'] = failed(lambda: str(f + g))
    out['sub'] = failed(lambda: str(f - g))
    out['mul'] = failed(lambda: str(f * g))
    out['gcd'] = failed(lambda: str(f.gcd(g)))
    out['lcm'] = failed(lambda: str(f.lcm(g)))
    out['cofactors'] = failed(lambda: ' '.join(map(str, f.cofactors(g))))
    out['cancel'] = failed(lambda: ' '.join(map(str, f.cancel(g))))
    out['div'] = failed(lambda: ' '.join(map(str, f.div(g))))
    out['exquo'] = failed(lambda: str((f * g).exquo(g)))
    return out


def check(rc: Any, e: Any, other: Any, gens: list[Any]) -> None:
    expected = python_report(e, other, gens)
    got = dict(rc.poly_report(e, other, gens))
    for key, value in expected.items():
        assert str(got.get(key)) == str(value), (key, e, other, gens)


def random_polynomial(rng: random.Random, variables: list[Any], params: list[Any]) -> Any:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    terms = []
    for _ in range(rng.randint(1, 4)):
        c: Any = rng.choice([Integer(rng.randint(-5, 5)), Rational(rng.randint(-3, 3), rng.randint(1, 3))])
        if params and rng.random() < 0.4:
            c = mul(c, rng.choice(params))
        factors = [c] + [pow(v, Integer(rng.randint(0, 3))) for v in variables if rng.random() < 0.7]
        terms.append(mul(*factors))
    return add(*terms)


@pytest.mark.parametrize("seed", range(40))
def test_random_polynomials(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.mul import mul
    rc = load("sympy_extras_rs.core")
    x, y, z, a, b = [Symbol(c) for c in 'xyzab']
    rng = random.Random(stable_seed("poly", seed))
    for _ in range(5):
        variables = [x, y, z][:rng.choice([1, 1, 2, 3])]
        params = [a, b][:rng.choice([0, 0, 1, 2])]
        common = random_polynomial(rng, variables, params)
        e = (mul(random_polynomial(rng, variables, params), common) if rng.random() < 0.5
             else random_polynomial(rng, variables, params))
        other = mul(random_polynomial(rng, variables, params), common)
        check(rc, e, other, rng.choice([[], variables]))


def test_edge_cases(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol
    from sympy_extras.functions import sin, sqrt
    rc = load("sympy_extras_rs.core")
    x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    cases: list[tuple[Any, Any, list[Any]]] = [
        (x**2 + 1, x + 1, [x, x]),
        (sin(x) + 1, x, [x]),
        (sin(x)*x + 1, x, []),
        (sqrt(x) + 1, x, []),
        (1/x + 1, x, [x]),
        (I*x + 1, x, [x]),
        (Integer(3), x, []),
        (Integer(0), x, [x]),
        (Rational(1, 2)*x**2, Integer(2)*x, [x]),
        ((x + a)**3, x + a, [x]),
        ((x*y)**2 + y, x*y, [x, y]),
        (a*x**2 + x, Integer(2)*x + a, [x]),
        (x**2 - 1, Integer(0), [x]),
        (x*y + 1, x + 1, [y]),
        ((2*x + 2)/3, x + 1, []),
    ]
    for e, other, gens in cases:
        check(rc, e, other, gens)
