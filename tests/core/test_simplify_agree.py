"""``simplify`` (with ``collect_abs``, ``inversecombine``,
``kroneckersimp``) of ``sympy_extras.simplify.simplify`` against its Rust
port: the cases of its tests and random expressions of trigonometric,
exponential, logarithmic, gamma and absolute value functions; the
outputs compared as strings (``NotImplementedError`` on the Rust side is
a skip, for the branches not ported: Piecewise conditions, hyper and
meijerg, the Bessel functions)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('simplify', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        return ('ok', str(f()))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _symbols() -> Any:
    from sympy_extras.core import Symbol
    return Symbol('x'), Symbol('y'), Symbol('p', positive=True), Symbol('n', integer=True)


def _cases() -> list[Any]:
    from sympy_extras.core import Integer, Rational, pi
    from sympy_extras.functions import Abs, binomial, cos, cosh, exp, factorial, gamma, log, sign, sin, sinh, sqrt, tan
    x, y, p, n = _symbols()
    return [sin(x)**2 + cos(x)**2, (x**3 + x**2 - x - 1)/(x**2 + 2*x + 1), gamma(x + 1)/gamma(x), log(Integer(2)) + log(Integer(3)),
            (x + 1)**2 - x**2 - 2*x, 1/x + 1/y, exp(x)*exp(y), sqrt(p**2), Abs(x)*Abs(y), sign(2*x), log(p**2)/2,
            cos(x)**4 - sin(x)**4, (exp(x) - exp(-x))/2, factorial(n + 1)/factorial(n), 1/(sqrt(Integer(2)) + 1),
            x/(x + y) + y/(x + y), tan(x)*cos(x), Abs(x + 1)/Abs(x**2 - 1), cosh(x)**2 - sinh(x)**2, binomial(n, 2),
            exp(2*log(x)), log(x*y) - log(x), p/4 + log(x**2/p) + gamma(pi*y) + pi/4 + 2, Integer(0), x - x,
            (x**2 - 1)/(x - 1) + Rational(1, 2), sqrt(x**2 + 2*x + 1), exp(log(x) + log(y))]


def _random(seed: int) -> list[Any]:
    from sympy_extras.core import Integer, Rational, pi
    from sympy_extras.functions import Abs, cos, exp, gamma, log, sin, sqrt, tan
    x, y, p, n = _symbols()
    rng = random.Random(seed)
    atoms = [x, y, p, Integer(2), Rational(1, 2), pi]
    funcs = [sin, cos, exp, log, sqrt, lambda a: a**2, lambda a: 1/a, tan, gamma, Abs]

    def rnd(d: int) -> Any:
        if d == 0 or rng.random() < .25:
            return rng.choice(atoms)
        r = rng.random()
        if r < .35:
            return rnd(d - 1) + rnd(d - 1)
        if r < .7:
            return rnd(d - 1)*rnd(d - 1)
        return rng.choice(funcs)(rnd(d - 1))
    return [rnd(3) for _ in range(25)]


def _compare(e: Any, **options: Any) -> tuple[Any, Any]:
    from sympy_extras.core import Integer, Rational, Symbol, oo
    from sympy_extras.core.logic import false, true
    from sympy_extras.simplify.simplify import simplify
    ratio = options.get('ratio', 1.7)
    r = oo if ratio == float('inf') else Rational(int(round(ratio*10)), 10)
    got = _outcome(lambda: _call('simplify', e, r, true if options.get('inverse') else false, false if options.get('doit') is False else true))
    expected = _outcome(lambda: simplify(e, **options))
    return got, expected


@pytest.mark.parametrize("index", range(len(_cases())))
def test_simplify(rust: Any, index: int) -> None:
    e = _cases()[index]
    for options in ({}, {'ratio': float('inf')}, {'inverse': True}, {'ratio': 1.0}):
        got, expected = _compare(e, **options)
        if got[0] == 'SKIP':
            continue
        assert got == expected, (e, options)


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_simplify_random(rust: Any, seed: int) -> None:
    different = []
    for e in _random(seed):
        got, expected = _compare(e)
        if got[0] != 'SKIP' and got != expected:
            different.append((e, got, expected))
    assert not different, different[:3]


def test_helpers(rust: Any) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import Abs, KroneckerDelta, acosh, asin, asinh, atan, cosh, erf, erfinv, exp, log, sin, sinh, tan
    from sympy_extras.simplify.simplify import collect_abs, inversecombine, kroneckersimp
    x, y, p, n = _symbols()
    i, j = Symbol('i'), Symbol('j')
    for e in [log(exp(x)) + exp(log(y)), atan(tan(x)), asinh(sinh(exp(log(x)))), erf(erfinv(x))*tan(atan(y)), sin(asin(x)) + acosh(cosh(x))]:
        assert _outcome(lambda: _call('inversecombine', e)) == _outcome(lambda: inversecombine(e)), e
    K = KroneckerDelta
    for e in [1 + K(Integer(0), i)*K(Integer(1), i), K(i, j)*K(Integer(0), i)*K(Integer(1), j), x*K(i, Integer(0))**2 + K(i, Integer(2))*K(i, Integer(1))]:
        assert _outcome(lambda: _call('kroneckersimp', e)) == _outcome(lambda: kroneckersimp(e)), e
    for e in [Abs(x + 1)/Abs(x**2 - 1), Abs(x)*Abs(y)*x, Abs(x)**2*Abs(y)**-1, Abs(x)*y, 1/Abs(x)]:
        assert _outcome(lambda: _call('collect_abs', e)) == _outcome(lambda: collect_abs(e)), e
