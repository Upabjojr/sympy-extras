"""ask by cases and by refutation (core.ask._ask_by_cases,
_ask_by_refutation) against SymPy's ask: no disagreement on random
propositions and assumptions, and the cases SymPy decided through the
bridge decided natively."""
from __future__ import annotations

import random

import sympy
from sympy.core.cache import clear_cache

from sympy.logic.boolalg import Boolean as SympyBoolean

from sympy_extras.core import Implies, Q, Symbol, ask
from sympy_extras.core.ask import _Facts, _ask_by_cases, _ask_by_refutation
from sympy_extras.core.logic import Boolean
from sympy_extras.interop import from_sympy


def test_cases_and_refutation() -> None:
    x, y = Symbol('x'), Symbol('y')
    assert ask(Q.nonzero(x), Q.negative(x) | Q.positive(x)) is True
    assert ask(Q.integer(x), Q.even(x) | Q.odd(x)) is True
    assert ask(Q.real(x), Q.negative(x) ^ Q.positive(x)) is True
    assert ask(Q.positive(x), Q.real(x) & Implies(Q.real(x), Q.positive(x))) is True
    assert ask(Implies(Q.positive(x), Q.real(x))) is True
    assert ask(Q.negative(x) & Q.positive(x)) is False
    assert ask(Q.rational(x), Q.real(x) & ~Q.irrational(x)) is True
    assert ask(Q.gt(x, y), x > y) is True
    # undecided where the cases disagree (and where SymPy is wrong: issue #25)
    assert _ask_by_cases(Q.even(x), Q.odd(x) | Q.zero(x)) is None
    assert _ask_by_cases(Q.real(x*y), Q.imaginary(y) & Q.real(x)) is None
    assert _ask_by_refutation(Q.real(x*y), Q.imaginary(y) & Q.real(x)) is None


def test_random_queries_agree_with_sympy() -> None:
    rng = random.Random(11)
    x, y = sympy.symbols('x y')
    preds = [sympy.Q.positive, sympy.Q.negative, sympy.Q.zero, sympy.Q.nonzero, sympy.Q.real, sympy.Q.integer,
             sympy.Q.even, sympy.Q.odd, sympy.Q.rational, sympy.Q.irrational, sympy.Q.nonnegative, sympy.Q.complex]

    def lit() -> SympyBoolean:
        p = rng.choice(preds)(rng.choice([x, y]))
        return ~p if rng.random() < 0.3 else p

    def formula(depth: int) -> SympyBoolean:
        if depth == 0 or rng.random() < 0.3:
            return lit()
        a, b = formula(depth - 1), formula(depth - 1)
        return rng.choice([a & b, a | b, sympy.Implies(a, b), a ^ b, ~a])

    decided = 0
    for _ in range(400):
        p, a = formula(1), formula(2)
        if a in (sympy.true, sympy.false) or p in (sympy.true, sympy.false):
            continue
        clear_cache()
        try:
            want = sympy.ask(p, a)
        except ValueError:
            continue
        native_p, native_a = from_sympy(p), from_sympy(a)
        assert isinstance(native_p, Boolean) and isinstance(native_a, Boolean)
        try:
            _Facts(native_a, native_p)
        except ValueError:
            continue
        got = _ask_by_cases(native_p, native_a)
        if got is None:
            got = _ask_by_refutation(native_p, native_a)
        if got is not None:
            decided += 1
            assert want is None or got == want, (p, a, got, want)
    assert decided > 30
