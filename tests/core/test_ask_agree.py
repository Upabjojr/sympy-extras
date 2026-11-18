"""``sympy_extras.core.ask`` against its Rust port, up to the last steps
of SymPy's ``ask`` (``assumptions.handlers.final_ask``, not ported yet:
patched to ``None`` on the Python side): the facts of the assumptions
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
    from sympy_extras.core import Eq, Q, Symbol
    x, y = Symbol('x'), Symbol('y')
    n = Symbol('n', integer=True)
    return [Q.positive(x), Q.positive(x + 1), Q.nonzero(x), Q.real(x*y), Q.integer(x + n), Q.even(2*n),
            Q.odd(2*n + 1), Q.negative(-x), Q.finite(1/x), Q.zero(x*y), Q.positive_infinite(x),
            Q.gt(x, 0), Q.lt(x + 1, x), Q.eq(x, x), Q.ne(x, 1), Q.is_true(x > -1), Q.is_true(Eq(x, y)),
            Q.positive(x) & Q.real(y), Q.positive(x) | Q.negative(x), ~Q.zero(x), Q.positive(x) >> Q.real(x),
            Q.rational(x) ^ Q.irrational(x), Q.prime(7), Q.composite(4), Q.positive(n) | Q.nonpositive(n),
            Q.extended_real(x)]


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
    monkeypatch.setattr(python_ask, '_final_ask', lambda prop, assumptions: None)
    assumptions = _assumptions()[index]
    for prop in _propositions():
        def got() -> Any:
            r = rc.logic_call('ask', [prop, assumptions])
            return None if isinstance(r, Tuple) else bool(r)
        assert outcome(got) == outcome(lambda: python_ask.ask(prop, assumptions)), (prop, assumptions)
