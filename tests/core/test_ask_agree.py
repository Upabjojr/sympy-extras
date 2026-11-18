"""``sympy_extras.core.ask``, ``sympy_extras.assumptions.handlers`` and
``sympy_extras.assumptions.satask`` against their Rust ports: the facts of the assumptions
about subexpressions, their closure and consistency, the answers of the
old assumptions on dummies carrying them, the relations, the
connectives, ``is_true``, the cases of disjunctive assumptions and the
refutation."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _propositions() -> list[Any]:
    from sympy_extras.core import Eq, I, Q, Symbol, pi, sqrt
    from sympy_extras.functions import Abs, atan, exp, factorial, log
    x, y = Symbol('x'), Symbol('y')
    n = Symbol('n', integer=True)
    return [Q.positive(x), Q.positive(x + 1), Q.nonzero(x), Q.real(x*y), Q.integer(x + n), Q.even(2*n),
            Q.odd(2*n + 1), Q.negative(-x), Q.finite(1/x), Q.zero(x*y), Q.positive_infinite(x),
            Q.gt(x, 0), Q.lt(x + 1, x), Q.eq(x, x), Q.ne(x, 1), Q.is_true(x > -1), Q.is_true(Eq(x, y)),
            Q.positive(x) & Q.real(y), Q.positive(x) | Q.negative(x), ~Q.zero(x), Q.positive(x) >> Q.real(x),
            Q.rational(x) ^ Q.irrational(x), Q.prime(7), Q.composite(4), Q.positive(n) | Q.nonpositive(n),
            Q.extended_real(x), Q.real(sqrt(x)*y), Q.positive(log(x)), Q.real(Abs(x) - 1), Q.integer(x**2),
            Q.rational(exp(x)), Q.imaginary(I*x), Q.even(x*n), Q.odd(x + 1), Q.prime(x**2), Q.composite(x),
            Q.nonzero(x**2 + 1), Q.zero(x*y), Q.negative(x*y), Q.positive(1 - x), Q.real(log(x)),
            Q.imaginary(log(x)), Q.complex(x + I), Q.positive(atan(x)), Q.integer(Abs(n)), Q.irrational(sqrt(x)),
            Q.real(x**y), Q.positive(factorial(n)), Q.nonnegative(x**2), Q.real(sqrt(2) + pi), Q.positive(pi - 3),
            Q.even(4), Q.prime(sqrt(49)), Q.imaginary(sqrt(-2)), Q.rational(sqrt(4)), Q.real(exp(I*pi))]


def _assumptions() -> list[Any]:
    from sympy_extras.core import Equivalent, Implies, Q, Symbol, true
    x, y = Symbol('x'), Symbol('y')
    p = Symbol('p', positive=True)
    return [true, Q.positive(x), Q.negative(x), Q.zero(x), Q.real(x) & Q.real(y), Q.positive(x) & Q.positive(y),
            Q.negative(x) | Q.positive(x), Q.even(x) | Q.odd(x), Q.negative(x) ^ Q.positive(x),
            Implies(Q.real(x), Q.positive(x)) & Q.real(x), Equivalent(Q.positive(x), Q.positive(y)) & Q.positive(y),
            Q.integer(x) & Q.positive(x), Q.positive_infinite(x), Q.positive(x) & Q.zero(x), ~Q.real(x),
            Q.is_true(x > 0), Q.positive(x) | Q.zero(y), Q.finite(x) & Q.nonzero(x), Q.irrational(x),
            Q.positive(p + 1), Q.imaginary(x), Q.zero(x) | Q.zero(y)]


@pytest.mark.parametrize("index", range(22))
def test_ask(rust: Any, index: int, monkeypatch: Any) -> None:
    import importlib
    python_ask = importlib.import_module('sympy_extras.core.ask')
    from sympy_extras.core import Tuple
    rc = load("sympy_extras_rs.core")
    assumptions = _assumptions()[index]
    for prop in _propositions():
        def got() -> Any:
            r = rc.logic_call('ask', [prop, assumptions])
            return None if isinstance(r, Tuple) else bool(r)
        expected = outcome(lambda: python_ask.ask(prop, assumptions))
        assert outcome(got) == expected, (prop, assumptions)


def test_predicates_consistent(rust: Any) -> None:
    from sympy_extras.assumptions.satask import predicates_consistent
    from sympy_extras.core import Q, Symbol
    from sympy_extras.functions import Abs
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    formulas = [Q.positive(x) & Q.integer(x), Q.positive(x) & Q.negative(x), Q.positive(x) & Q.positive(y) & Q.negative(x + y),
                Q.even(x) & Q.odd(x), Q.prime(x) & Q.composite(x), Q.zero(x*y) & Q.nonzero(x) & Q.nonzero(y),
                Q.negative(Abs(x)), Q.real(x) & Q.imaginary(x) & Q.nonzero(x), Q.rational(x) & Q.irrational(x),
                Q.positive(x**2) | Q.zero(x), x > 1, Q.gt(x, 1) & Q.lt(x, 0), Q.even(2) , Q.odd(2), Q.positive(-1),
                Q.integer(x) & Q.integer(y) & Q.odd(x*y) & Q.even(x)]
    for f in formulas:
        assert bool(rc.logic_call('satask', [f])) == predicates_consistent(f), f
