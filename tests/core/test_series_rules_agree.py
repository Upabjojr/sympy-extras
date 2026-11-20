"""The series and leading-term rules of the special functions of the Rust
core (``functions/series_rules.rs``, ``functions/leading_terms.rs``)
against ``sympy_extras.functions._series_rules`` and ``_leading_terms``:
``nseries`` and ``as_leading_term`` of the functions with a rule of their
own at arguments vanishing, finite and singular at ``x = 0``."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from ..conftest import load


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', str(r))


def functions() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.functions import (Abs, Chi, Ci, Ei, LambertW, Shi, Si, besseli, besselj, besselk, bessely,
                                        ceiling, elliptic_e, elliptic_k, erf, erfc, erfi, expint, factorial, floor,
                                        frac, fresnelc, fresnels, gamma, loggamma, polylog, sign)
    x = Symbol('x')
    args = [x, 2*x, x**2, x + Rational(1, 2), -x, x - 1]
    found: list[Any] = []
    for a in args:
        found += [Abs(a), sign(a), gamma(a), loggamma(a), Ei(a), Si(a), Ci(a), Shi(a), Chi(a), erf(a), erfc(a),
                  erfi(a), fresnels(a), fresnelc(a), floor(a), ceiling(a), frac(a), LambertW(a), factorial(a),
                  elliptic_k(a), elliptic_e(a), expint(Integer(1), a), expint(Integer(2), a), polylog(2, a),
                  polylog(3, a), besselj(0, a), besselj(1, a), besseli(0, a), besseli(Rational(1, 2), a),
                  bessely(1, a), besselk(0, a), besselk(1, a), besselk(Rational(1, 3), a)]
    return found


@pytest.mark.parametrize('n', [1, 3, 5])
def test_nseries(rust: Any, n: int) -> None:
    from sympy_extras.core import Integer, Symbol
    rc = load("sympy_extras_rs.core")
    x = Symbol('x')
    compared = 0
    for f in functions():
        got = outcome(lambda: rc.logic_call('series', [Symbol('nseries'), f, x, Integer(n)]))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: f.nseries(x, n=n)), (f, n)
        compared += 1
    assert compared > 50


def test_leading_terms(rust: Any) -> None:
    from sympy_extras.core import Symbol, Tuple
    rc = load("sympy_extras_rs.core")
    x = Symbol('x')
    compared = 0
    for f in functions():
        got = outcome(lambda: rc.logic_call('series', [Symbol('leading'), f, Tuple(x)]))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: f.as_leading_term(x)), f
        compared += 1
    assert compared > 50
