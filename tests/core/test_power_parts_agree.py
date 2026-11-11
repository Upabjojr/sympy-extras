"""The real and imaginary parts of powers (rational exponents by the
modulus and ``atan2``, powers of ``E``, the expanded fallback) and of
logarithms (``log(|z|) + I*arg(z)``), the logarithms of algebraic complex
numbers at the angles of SymPy's arctangent table, and ``ratsimp``, in
the Rust core against ``sympy_extras``, compared by structure and repr."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import shape
from ..polys.test_polytools_functions_agree import compare, random_polynomial


def same(got: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        return all(same(g, e) for g, e in zip(got, expected))
    return bool(shape(got) == shape(expected) and repr(got) == repr(expected))


def tangents() -> list[Any]:
    from sympy_extras.core import Integer, sqrt
    s2, s3, s5 = sqrt(2), sqrt(3), sqrt(5)
    return [s3, Integer(1), sqrt(5 - 2*s5), sqrt(5 + 2*s5), s3/3, s2 - 1, s2 + 1, sqrt(1 - 2*s5/5),
            sqrt(1 + 2*s5/5), 2 - s3, 2 + s3, Integer(2), s2]


def test_log_of_angles(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    from sympy_extras.functions import log
    rc = load("sympy_extras_rs.core")
    p = Symbol('p', positive=True)
    compared = 0
    for t in tangents():
        for real in (Integer(1), Integer(-1), Rational(3, 2), sqrt(2)):
            for sign in (1, -1):
                for scale in (Integer(1), Integer(3), p):
                    z = scale*(real + sign*t*real*I)
                    expected = log(z)
                    try:
                        got = rc.log(z)
                    except NotImplementedError:
                        continue
                    assert same(got, expected), (z, got, expected)
                    compared += 1
    assert compared > 200


def test_real_imaginary_parts(rust: Any) -> None:
    from sympy_extras.core import E, I, Integer, Rational, Symbol, pi, sqrt
    from sympy_extras.functions import exp, log
    rc = load("sympy_extras_rs.core")
    x, y, p = Symbol('x'), Symbol('y'), Symbol('p', positive=True)
    r = Symbol('r', real=True)
    bases = [1 + I, -8 + 0*I, Integer(-8), 2 - 3*I, I, x, r, x + I*y, p*I, -p, sqrt(2) - I, 1 + sqrt(3)*I]
    exponents = [Rational(1, 2), Rational(1, 3), Rational(-2, 3), Rational(5, 2), x, I, r, 1 + I]
    nodes = [b**k for b in bases for k in exponents] + [E**(x + I*y), E**(I*pi/3), E**(r + 2*I), exp(I*x)]
    nodes += [log(z) for z in (1 + I, -2, x, r, p*I, 3 - 4*I, x + I*y, -p)]
    compared = 0
    for e in nodes:
        expected = e.as_real_imag()
        try:
            got = rc.as_real_imag(e)
        except NotImplementedError:
            continue
        assert same(got, expected), (e, got, expected)
        compared += 1
    assert compared > 80


@pytest.mark.parametrize("seed", range(30))
def test_ratsimp(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.simplify.ratsimp import ratsimp
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("ratsimp", seed))
    x, y = Symbol('x'), Symbol('y')
    compared = 0
    for _ in range(4):
        variables = rng.choice([[x], [x, y]])
        e = add(mul(random_polynomial(rng, variables), pow(random_polynomial(rng, variables), Integer(-1))),
                mul(random_polynomial(rng, variables), pow(random_polynomial(rng, variables), Integer(-1))))
        compared += compare(rc, "ratsimp", lambda: ratsimp(e), [e], [])
    assert compared >= 2
