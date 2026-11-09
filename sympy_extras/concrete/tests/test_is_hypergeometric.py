"""The native ``Expr.is_hypergeometric`` against SymPy's on fixed and
random terms in ``k``."""

from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Symbol
from sympy_extras.interop import expr_from_sympy

sk, sn, sx = sympy.symbols('k n x')
k = Symbol('k')

FIXED: list[sympy.Expr] = [
    sympy.factorial(sk), sympy.binomial(sn, sk), sympy.binomial(2*sk, sk)/4**sk, sympy.gamma(sk + sympy.Rational(1, 2)),
    sympy.rf(sn, sk)/sympy.factorial(sk), sympy.ff(sx, sk), 2**sk*sk**3, sympy.exp(sk), sympy.exp(sk**2), sympy.sin(sk),
    sympy.log(sk), sympy.sqrt(sk), sympy.factorial(sk**2), sympy.harmonic(sk), sympy.fibonacci(sk), sk**sk, sympy.Integer(5),
    sympy.factorial(2*sk)/sympy.factorial(sk)**2, sympy.Piecewise((sk, sk > 0), (1, True)), 1/(sk**2 + 1), sympy.cos(sympy.pi*sk),
    (-1)**sk*sympy.binomial(sn, sk)**2, sympy.factorial(sk)*sympy.sin(sx), sympy.catalan(sk), sympy.gamma(sk)**2/sympy.gamma(2*sk),
]


@pytest.mark.parametrize('e', FIXED, ids=str)
def test_fixed(e: sympy.Expr) -> None:
    assert expr_from_sympy(e).is_hypergeometric(k) == e.is_hypergeometric(sk)


def test_random_products() -> None:
    rng = random.Random(20261109)
    factors = [sympy.factorial(sk), sympy.factorial(2*sk + 1), sympy.binomial(sn, sk), sympy.gamma(sk + sn), 3**sk, sk + 2,
               sympy.rf(sx, sk), sympy.exp(sk), sk**2 + sn, sympy.sin(sk), sympy.log(sk + 1), sympy.Rational(-1, 2)**sk,
               sympy.sqrt(sk + 1), sympy.harmonic(sk)]
    for _ in range(60):
        e = sympy.Integer(1)
        for _ in range(rng.randint(1, 3)):
            e = e*rng.choice(factors)**rng.choice([1, 1, -1, 2])
        assert expr_from_sympy(e).is_hypergeometric(k) == e.is_hypergeometric(sk), e
