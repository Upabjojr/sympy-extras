"""The LaTeX of the objects which are not functions
(``printing/latex_objects.rs``) against ``sympy_extras.printing.latex``:
piecewise expressions, sums, products, integrals, sets, quantifiers,
limits, order terms, accumulation bounds; the nodes not ported yet are a
skip on the Rust side."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_latex_agree import SETTINGS, _outcome


def _cases() -> list[Any]:
    from sympy_extras.assumptions.quantifiers import exists, forall
    from sympy_extras.concrete.expr_with_limits import product_, sum_
    from sympy_extras.core import O, Eq, Integer, Rational, Symbol, oo, pi
    from sympy_extras.core.accumbounds import AccumBounds
    from sympy_extras.core.limits import Limit
    from sympy_extras import functions as f
    from sympy_extras.integrals.integral import Integral
    from sympy_extras.sets import (Complement, ConditionSet, Contains, FiniteSet, ImageSet, Intersection, Interval,
                                   PowerSet, ProductSet, Union)
    from sympy_extras.sets.fancysets import Complexes, Integers, Naturals, Naturals0, Rationals
    from sympy_extras.sets.sets import EmptySet, Reals, UniversalSet
    from sympy_extras.core.containers import Tuple
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    k, n = Symbol('k', integer=True), Symbol('n', integer=True)
    one, zero = Integer(1), Integer(0)
    pw = f.Piecewise((x, x > 0), (-x, True))
    pw2 = f.Piecewise((x**2, x < 1), (x, x < 2))
    return [pw, pw2, pw + 1, 2*pw, pw**2,
            sum_(k, (k, one, n)), sum_(k**2/(k + 1), (k, one, oo)), sum_(x*k, (k, zero, n), (n, one, z)),
            2*sum_(k, (k, one, n)), sum_(k, (k, one, n))*x, product_(k, (k, one, n)), product_(1 + 1/k, (k, one, n)),
            Integral(x**2, x), Integral(f.sin(x), x, zero, pi), Integral(f.exp(-x**2), x, -oo, oo),
            2*Integral(x, x) + 1, Integral(x, x)*y, Integral(Integral(x*y, x), y),
            Interval(zero, one), Interval(zero, one, True, False), Interval(-oo, oo), Interval(one, oo, False, True),
            FiniteSet(one, x, Integer(2)), FiniteSet(), Union(Interval(zero, one), Interval(Integer(2), Integer(3))),
            Intersection(Interval(zero, Integer(2)), FiniteSet(x)),
            Complement(Reals, FiniteSet(zero)), ProductSet(Interval(zero, one), Interval(zero, one)),
            ProductSet(Interval(zero, one), FiniteSet(one)), EmptySet, UniversalSet, Naturals, Naturals0, Integers,
            Rationals, Reals, Complexes, ImageSet(Tuple(x), x**2, Integers), ConditionSet(x, x > 1, Reals),
            ConditionSet(x, Eq(x**2, 1), UniversalSet), PowerSet(FiniteSet(one, Integer(2))),
            Contains(x, Interval(zero, one)), forall(x, x**2 >= 0), exists((x, y), Eq(x, y)),
            Limit(f.sin(x)/x, x, zero), Limit(1/x, x, zero, '-'), Limit(x, x, oo), Limit(f.exp(x), x, one, '+-'),
            O(x**3), O(x, (x, oo)), O(x*y), x + O(x**2), AccumBounds(-one, one), AccumBounds(zero, Rational(1, 2))]


@pytest.mark.parametrize("index", range(len(_cases())))
def test_latex_objects(rust: Any, index: int) -> None:
    from sympy_extras.printing.latex import latex
    rc = load("sympy_extras_rs.core")
    e = _cases()[index]
    for settings in SETTINGS:
        got = _outcome(lambda: rc.latex(e, settings))
        if got[0] == 'SKIP':
            continue
        assert got == _outcome(lambda: latex(e, **settings)), (e, settings)
