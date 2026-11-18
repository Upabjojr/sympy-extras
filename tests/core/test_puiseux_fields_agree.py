"""The number fields of ``sympy_extras.polys.puiseux`` against their Rust
port: ``field_from_polynomial`` (the generator as the largest real root,
in radicals when trivial), ``field_of`` (radicals, ``I``, an
``IndexedRoot``), ``adjoin_root`` over the rationals and over quadratic
fields (the primitive element, the image of the old generator, the
minimal polynomial of the root, ``in_subfield``) and ``factor_over``."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def rationals(values: list[Fraction]) -> Any:
    from sympy_extras.core import Rational
    from sympy_extras.core.containers import Tuple
    return Tuple(*[Rational(v.numerator, v.denominator) for v in values])


def field_rs(K: Any) -> Any:
    return [rationals(list(K.minpoly)), K.alpha_expr]


def py_field(K: Any) -> Any:
    return (str([Fraction(c) for c in K.minpoly]), str(K.alpha_expr))


def rs_field(r: Any) -> Any:
    return (str([Fraction(str(c)) for c in r.args[0].args]), str(r.args[1]))


def numbers(r: Any) -> list[Fraction]:
    return [Fraction(str(c)) for c in r.args]


POLYNOMIALS = [[1, 0, -2], [1, 0, 1], [1, 0, 0, -2], [1, 0, -1, -1], [1, -1, -1], [2, 0, 3], [1, 0, 0, 0, 2],
               [1, 0, -4, 2], [3, 1], [1, 2, 3, 4], [1, 0, 0, -3, 1]]


@pytest.mark.parametrize("coefficients", POLYNOMIALS)
def test_field_from_polynomial(rust: Any, coefficients: list[int]) -> None:
    from sympy_extras.polys.puiseux import field_from_polynomial
    rc = load("sympy_extras_rs.core")
    values = [Fraction(c) for c in coefficients]
    got = outcome(lambda: rs_field(rc.logic_call('puiseux_field_from_polynomial', [rationals(values)])))
    assert got == outcome(lambda: py_field(field_from_polynomial(values)))


def test_field_of(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    from sympy_extras.polys.cad.indexedroot import IndexedRoot, bound_variable
    from sympy_extras.polys.puiseux import field_of
    rc = load("sympy_extras_rs.core")
    T = bound_variable('T')
    for point in [Rational(1, 2), 1 + sqrt(2), I, sqrt(3)*I + 1, IndexedRoot(T**3 - T - 1, T, 0), 2**Rational(1, 3)]:
        def got() -> Any:
            r = rc.logic_call('puiseux_field_of', [point])
            return rs_field(r.args[0]), numbers(r.args[1])

        def expected() -> Any:
            K, c = field_of(point)
            return py_field(K), list(c)
        assert outcome(got) == outcome(expected), point


@pytest.mark.parametrize("seed", range(30))
def test_adjoin_root(rust: Any, seed: int) -> None:
    from sympy_extras.core import sqrt
    from sympy_extras.core.containers import Tuple
    from sympy_extras.polys.puiseux import (RATIONALS, adjoin_root, factor_over, field_of, in_subfield,
                                            minimal_polynomial_of, generator)
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("puiseux fields", seed))
    K = RATIONALS if rng.random() < 0.4 else field_of(rng.choice([sqrt(2), sqrt(-3), sqrt(5)]))[0]
    d = rng.choice([2, 2, 3])
    coefficients = [[Fraction(1)]] + [K.from_rational(Fraction(rng.randint(-3, 3))) for _ in range(d)]
    if K.degree > 1 and rng.random() < 0.5:
        coefficients[-1] = K.add(coefficients[-1], K.alpha)

    def factors_py() -> Any:
        return [([list(c) for c in f], m) for f, m in factor_over(K, coefficients)]

    def factors_rs() -> Any:
        r = rc.logic_call('puiseux_factor_over', [*field_rs(K), Tuple(*[rationals(c) for c in coefficients])])
        return [([numbers(c) for c in f.args[0].args], int(f.args[1])) for f in r.args]
    assert outcome(factors_rs) == outcome(factors_py)
    factors = factor_over(K, coefficients)
    psi = max(factors, key=lambda pair: len(pair[0]))[0]

    def adjoined_py() -> Any:
        L, into, u = adjoin_root(K, psi)
        back = in_subfield(into(generator(K)), L, into)
        return (py_field(L), None if into.image is None else list(into.image), list(u),
                list(minimal_polynomial_of(u, L)), None if back is None else list(back))

    def adjoined_rs() -> Any:
        r = rc.logic_call('puiseux_adjoin_root', [*field_rs(K), Tuple(*[rationals(c) for c in psi])])
        image = None if str(r.args[1]) == 'False' else numbers(r.args[1])
        back = None if str(r.args[4]) == 'False' else numbers(r.args[4])
        return (rs_field(r.args[0]), image, numbers(r.args[2]), numbers(r.args[3]), back)
    assert outcome(adjoined_rs) == outcome(adjoined_py), (K, psi)
