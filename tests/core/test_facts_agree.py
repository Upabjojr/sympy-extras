"""``sympy_extras.assumptions.facts`` against its Rust port: the normal
form (memberships of intervals, finite sets, unions, intersections and
complements as relations, relations canonical with their symbolic
denominators cleared, the connectives), the translations to predicates
of ``Q`` and to polynomial relations for the CAD, and :class:`Facts`
(predicates, real and integer variables, the polynomial part, domains)."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _formulas() -> list[Any]:
    from sympy_extras.assumptions.facts import element
    from sympy_extras.core import ITE, Eq, Implies, Ne, Not, Q, Symbol, oo
    from sympy_extras.functions import exp
    from sympy_extras.sets import Complement, FiniteSet, Integers, Intersection, Interval, Naturals, Reals, Union
    x, y, z, a = (Symbol(n) for n in 'xyza')
    return [x > 0, element(x, Reals), element(x, Naturals), element(x, Interval(0, 1)),
            element(x, Interval(0, 1, True, True)), element(x, Interval(0, oo)), element(x, FiniteSet(1, 2)),
            element(x, Union(FiniteSet(1), Interval(2, 3, True, True))), element(x, Complement(Reals, FiniteSet(0))),
            element(x, Complement(Integers, FiniteSet(0))), Not(element(x, Integers)), ITE(x > 0, y > 0, z > 0),
            Implies(x > 0, element(y, Interval(0, 1))), Q.positive(x), x/a > 1, Eq(x/a, 1), Ne(x/(a**2 + 1), 2),
            x/a >= 0, x/a <= 0, x/a < 3, exp(-x) - x**2 < 0, 2 < x, Eq(x, 0), Ne(x, 0), x > y,
            element(x, Integers) & (x > 0), (x > 2) & element(y, Integers) & (z**3 > x*y), x*y < 1, Eq(x**2, 2),
            x >= -1, x <= 3, Eq(x, Symbol('w')**2 + 1), Eq(x, 3) | (y > 0)]


@pytest.mark.parametrize("index", range(33))
def test_normalize_and_translations(rust: Any, index: int) -> None:
    from sympy_extras.assumptions.facts import normalize, to_polynomial, to_predicates
    from sympy_extras.core import Symbol, Tuple
    rc = load("sympy_extras_rs.core")
    f = _formulas()[index]

    def call(*args: Any) -> Any:
        r = rc.logic_call('facts', [Symbol(args[0]), *args[1:]])
        return None if isinstance(r, Tuple) and not r.args else r
    assert outcome(lambda: call('normalize', f)) == outcome(lambda: normalize(f)), f
    n = normalize(f)
    assert outcome(lambda: call('to_predicates', n)) == outcome(lambda: to_predicates(n)), n
    real = sorted(n.free_symbols, key=lambda s: s.name)
    assert outcome(lambda: call('to_polynomial', n, Tuple(*real))) == outcome(lambda: to_polynomial(n, set(real))), n


@pytest.mark.parametrize("index", range(33))
def test_facts(rust: Any, index: int) -> None:
    from sympy_extras.assumptions.facts import Facts
    from sympy_extras.core import Symbol, Tuple
    from sympy_extras.sets import Integers, Reals
    rc = load("sympy_extras_rs.core")
    f = _formulas()[index]
    for domain in (None, Reals, Integers):
        def got() -> Any:
            r = rc.logic_call('facts', [Symbol('facts'), Tuple(f), Tuple() if domain is None else domain, Tuple()])
            formula, conjuncts, predicates, real, integer, polynomial = r.args
            return [str(formula), [str(c) for c in conjuncts.args], str(predicates), sorted(str(s) for s in real.args),
                    sorted(str(s) for s in integer.args), str(polynomial)]

        def expected() -> Any:
            F = Facts([f], domain=domain)
            return [str(F.formula), [str(c) for c in F.conjuncts], str(F.predicates), sorted(str(s) for s in F.real),
                    sorted(str(s) for s in F.integer), str(F.polynomial)]
        assert outcome(got) == outcome(expected), (f, domain)
