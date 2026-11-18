"""``sympy_extras.core.refine`` against its Rust port: the rules of the
classes (powers of negated bases, ``exp`` of rational multiples of
``pi*I``, Booleans decided by ``ask``) and the handlers of ``Abs``,
powers, ``atan2``, ``re``, ``im``, ``arg`` and ``sign``, under predicate
assumptions."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _expressions() -> list[Any]:
    from sympy_extras.core import I, Q, Rational, Symbol, pi, sqrt
    from sympy_extras.functions import Abs, arg, atan2, exp, im, re, sign
    x, y, n = Symbol('x'), Symbol('y'), Symbol('n')
    return [Abs(x), Abs(x - 1), Abs(x*y), Abs(x) + Abs(y), sqrt(x**2), (x**2)**Rational(1, 2), (-x)**n, (-x)**(2*n),
            (-1)**(n + 1), (-1)**(2*n + 3), (-1)**(n + Rational(1, 2)), exp(I*pi*n), exp(2*I*pi*n), exp(I*pi*n/2),
            atan2(y, x), re(x), im(x), re(x*y), im(x + I*y), arg(x), sign(x), sign(x*y), sign(I*x), Abs(x)**2,
            Q.positive(x + 1), Q.real(x) & Q.positive(y), (x**3)**Rational(1, 3), Abs(sqrt(x)*y), re(exp(I*x))]


def _assumptions() -> list[Any]:
    from sympy_extras.core import Q, Symbol, true
    x, y, n = Symbol('x'), Symbol('y'), Symbol('n')
    return [true, Q.positive(x), Q.negative(x), Q.real(x), Q.positive(x) & Q.negative(y), Q.integer(n),
            Q.even(n), Q.odd(n), Q.real(x) & Q.real(y), Q.zero(y) & Q.negative(x), Q.imaginary(x),
            Q.positive(y) & Q.zero(x), Q.negative(x) & Q.negative(y)]


@pytest.mark.parametrize("index", range(13))
def test_refine(rust: Any, index: int) -> None:
    from sympy_extras.core.refine import refine
    rc = load("sympy_extras_rs.core")
    assumptions = _assumptions()[index]
    for e in _expressions():
        assert outcome(lambda: rc.logic_call('refine', [e, assumptions])) == \
            outcome(lambda: refine(e, assumptions)), (e, assumptions)
