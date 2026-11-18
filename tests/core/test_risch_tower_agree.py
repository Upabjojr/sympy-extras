"""``sympy_extras.integrals.risch.tower`` against its Rust port: the
canonical form of :class:`RationalFunction` (the generators in SymPy's
order, the algebraic constants ``I`` and ``sqrt(p)`` reduced and cleared
from the denominators, the cancellation over ``ZZ``) and the dense
arithmetic of :class:`TowerPoly` over it (division, gcd, the cancelled
pair of SymPy's ``Poly.cancel``, resultants and the subresultant PRS,
square-free decompositions, extended Euclid, substitutions)."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _symbols() -> Any:
    from sympy_extras.core import Symbol
    return Symbol('x'), Symbol('t'), Symbol('t1'), Symbol('a')


def _atoms() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, pi, sqrt
    from sympy_extras.functions import exp, log
    x, t, t1, a = _symbols()
    return [x, t, t1, a, I, sqrt(2), sqrt(3), sqrt(6), Integer(2), Integer(-3), Rational(1, 2), pi,
            exp(x), exp(2*x), log(x), log(3), sqrt(-5), sqrt(Rational(3, 4))]


def _random_expr(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import Integer
    atoms = _atoms()
    if depth == 0:
        return rng.choice(atoms)
    a, b = _random_expr(rng, depth - 1), _random_expr(rng, depth - 1)
    op = rng.randrange(5)
    if op == 0:
        return a + b
    if op == 1:
        return a*b
    if op == 2:
        return a - b*Integer(rng.randint(-3, 3))
    if op == 3:
        return a/(b + Integer(rng.randint(1, 3)))
    return a**rng.randint(0, 3) + b


def _elements(count: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    return [_random_expr(rng, rng.randint(1, 2)) for _ in range(count)]


def _rf_result(r: Any) -> tuple[str, tuple[str, ...]]:
    return (str(r.as_expr()), tuple(str(g) for g in r.gens))


def _rust_rf(rc: Any, op: str, *args: Any) -> Callable[[], Any]:
    from sympy_extras.core import Symbol

    def got() -> Any:
        r = rc.logic_call('tower_rf', [Symbol(op)] + list(args))
        return (str(r.args[0]), tuple(str(g) for g in r.args[1].args))
    return got


@pytest.mark.parametrize("seed", range(12))
def test_rational_function(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer
    from sympy_extras.integrals.risch.tower import RationalFunction as RF
    rc = load("sympy_extras_rs.core")
    x, t, _, _ = _symbols()
    es = _elements(8, 1000 + seed)
    for e in es:
        assert outcome(_rust_rf(rc, 'from_expr', e)) == outcome(lambda: _rf_result(RF.from_expr(e))), e
    for e, f in zip(es, es[1:]):
        for op, fn in [('add', lambda p, q: p + q), ('sub', lambda p, q: p - q), ('mul', lambda p, q: p*q),
                       ('div', lambda p, q: p/q)]:
            assert outcome(_rust_rf(rc, op, e, f)) == \
                outcome(lambda: _rf_result(fn(RF.from_expr(e), RF.from_expr(f)))), (op, e, f)
        assert outcome(_rust_rf(rc, 'subs', e, x, f)) == \
            outcome(lambda: _rf_result(RF.from_expr(e).subs(x, RF.from_expr(f)))), (e, f)
    for e in es:
        assert outcome(_rust_rf(rc, 'diff', e, x)) == outcome(lambda: _rf_result(RF.from_expr(e).diff(x))), e
        assert outcome(_rust_rf(rc, 'conjugate_i', e)) == \
            outcome(lambda: _rf_result(RF.from_expr(e).conjugate_i())), e
        assert outcome(_rust_rf(rc, 'pow', e, Integer(-2))) == outcome(lambda: _rf_result(RF.from_expr(e)**-2)), e
        assert outcome(_rust_rf(rc, 'numerator', e)) == outcome(lambda: _rf_result(RF.from_expr(e).numerator())), e
        assert outcome(_rust_rf(rc, 'denominator', e)) == \
            outcome(lambda: _rf_result(RF.from_expr(e).denominator())), e


def _random_poly(rng: random.Random, t: Any) -> Any:
    """A polynomial in ``t`` with small coefficients: one of the symbols,
    a kernel, plus an integer (the algebraic constants make the Euclidean
    remainder sequences too slow for a test, on both sides)."""
    from sympy_extras.core import Integer
    from sympy_extras.functions import exp
    x, _, _, a = _symbols()
    choices = [x, a, exp(x), Integer(1), Integer(2), x/2, Integer(3)]
    total = Integer(0)
    for k in range(rng.randint(0, 2) + 1):
        c = rng.choice(choices) + Integer(rng.randint(-2, 2))
        total = total + c*t**k
    return total


def _poly_cases(seed: int) -> list[tuple[Any, Any]]:
    rng = random.Random(seed)
    _, t, _, _ = _symbols()
    cases = []
    for _ in range(6):
        common = _random_poly(rng, t)
        cases.append((common*_random_poly(rng, t), common*_random_poly(rng, t)**rng.randint(1, 2)))
    return cases


def _rust_poly(rc: Any, op: str, *args: Any) -> Callable[[], Any]:
    from sympy_extras.core import Symbol

    def got() -> Any:
        r = rc.logic_call('tower_poly', [Symbol(op)] + list(args))
        return tuple(str(v) for v in r.args)
    return got


@pytest.mark.parametrize("seed", range(12))
def test_tower_poly(rust: Any, seed: int) -> None:
    from sympy_extras.integrals.risch.tower import RationalFunction as RF, TowerPoly as TP
    rc = load("sympy_extras_rs.core")
    x, t, _, _ = _symbols()
    binary: dict[str, Callable[[Any, Any], Any]] = {
        'gcd': lambda p, q: (p.gcd(q),), 'lcm': lambda p, q: (p.lcm(q),), 'exquo': lambda p, q: (p.exquo(q),),
        'invert': lambda p, q: (p.invert(q),), 'div': lambda p, q: p.div(q), 'cancel': lambda p, q: p.cancel(q),
        'half_gcdex': lambda p, q: p.half_gcdex(q), 'gcdex': lambda p, q: p.gcdex(q),
        'resultant': lambda p, q: (p.resultant(q),),
        'resultant_prs': lambda p, q: (lambda r: (r[0],) + tuple(r[1]))(p.resultant_prs(q)),
    }

    def shown(values: Any) -> tuple[str, ...]:
        return tuple(str(v.as_expr()) for v in values)
    for f, g in _poly_cases(2000 + seed):
        for op, fn in binary.items():
            assert outcome(_rust_poly(rc, op, f, g, t)) == \
                outcome(lambda: shown(fn(TP.from_expr(f, t), TP.from_expr(g, t)))), (op, f, g)
        for e in (f, g):
            P = lambda: TP.from_expr(e, t)  # noqa: E731
            assert outcome(_rust_poly(rc, 'from_expr', e, t)) == outcome(lambda: shown([P()])), e
            assert outcome(_rust_poly(rc, 'monic', e, t)) == outcome(lambda: shown([P().monic()])), e
            assert outcome(_rust_poly(rc, 'degree', e, t, x)) == outcome(lambda: (str(P().degree(x)),)), e
            assert outcome(_rust_poly(rc, 'as_element', e, t)) == outcome(lambda: shown([P().as_element()])), e
            assert outcome(_rust_poly(rc, 'diff', e, t)) == outcome(lambda: shown([P().diff()])), e
            assert outcome(_rust_poly(rc, 'diff', e, t, x)) == outcome(lambda: shown([P().diff(x)])), e
            assert outcome(_rust_poly(rc, 'sqf_list_include', e, t)) == \
                outcome(lambda: tuple('(%s, %d)' % (p.as_expr(), k) for p, k in P().sqf_list_include())), e
            for v in (x + 1, t**2 + x, x/(t + 1)):
                assert outcome(_rust_poly(rc, 'subs', e, t, x, v)) == \
                    outcome(lambda: shown([P().subs(x, RF.from_expr(v))])), (e, v)
            assert outcome(_rust_poly(rc, 'subs', e, t, t, x + 1)) == \
                outcome(lambda: shown([P().subs(t, RF.from_expr(x + 1))])), e
            assert outcome(_rust_poly(rc, 'compose_shift', e, t, x - 2)) == \
                outcome(lambda: shown([P().compose_shift(RF.from_expr(x - 2))])), e

            def as_poly() -> Any:
                q = P().as_poly(x)
                return () if q is None else (str(q.as_expr()), str(q.gen))
            assert outcome(_rust_poly(rc, 'as_poly', e, t, x)) == outcome(as_poly), e
