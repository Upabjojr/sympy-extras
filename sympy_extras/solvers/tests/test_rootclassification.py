"""Root classification: the rule of signs against the real roots SymPy
counts, on numerical polynomials (with multiple roots and vanishing
minors) and at values of the parameters on and off the boundaries."""
from __future__ import annotations

import random
from typing import Optional

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, false, true
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.solvers.rootclassification import discrimination_sequence, real_root_classification, signature_from_signs

x, a, b, c, p, q = (Symbol(s) for s in 'xabcpq')
sx = sympy.Symbol('x')


def _distinct_real_roots(f: sympy.Expr) -> Optional[int]:
    if f == 0:
        return None
    if not f.has(sx):
        return 0
    return len(set(sympy.real_roots(sympy.Poly(f, sx))))


def _sign(e: Expr) -> int:
    assert e.is_number
    return 1 if e.is_positive else (-1 if e.is_negative else 0)


def test_rule_of_signs_on_numerical_polynomials() -> None:
    rnd = random.Random(171)
    zeros_seen = 0
    for trial in range(120):
        # products of small factors: multiple roots and complex pairs, so that minors vanish
        f: Expr = Integer(rnd.choice([1, -1, 2]))
        for _ in range(rnd.randint(1, 4)):
            kind = rnd.randint(0, 3)
            if kind == 0:
                f = f*(x - rnd.randint(-2, 2))
            elif kind == 1:
                f = f*(x - rnd.randint(-1, 1))**2
            elif kind == 2:
                f = f*(x**2 + rnd.randint(0, 2)*x + rnd.randint(1, 3))
            else:
                f = f*(x**2 - rnd.randint(1, 3))
        f = f.expand()
        signs = [_sign(d) for d in discrimination_sequence(f, x)]
        zeros_seen += 0 in signs[:-1]
        expected = _distinct_real_roots(sympy_expr(to_sympy(f)))
        assert signature_from_signs(signs) == expected, (f, signs)
        assert real_root_classification(f, x) == [(true, expected)]
    assert zeros_seen > 10


def _evaluate(cases: list[tuple[Boolean, Optional[int]]], values: dict[Symbol, Expr]) -> list[Optional[int]]:
    found: list[Optional[int]] = []
    for condition, count in cases:
        held: Boolean = condition
        for symbol, value in values.items():
            substituted = held.subs(symbol, value)
            assert isinstance(substituted, Boolean)
            held = substituted
        assert held is true or held is false, held
        if held is true:
            found.append(count)
    return found


def test_parametric_families_against_counts() -> None:
    rnd = random.Random(172)
    families: list[tuple[Expr, list[Symbol]]] = [
        (x**2 + b*x + c, [b, c]),
        (a*x**2 + b*x + c, [a, b, c]),
        (x**3 + p*x + q, [p, q]),
        (x**3 + a*x**2 + b*x + 1, [a, b]),
        (a*x**3 + x**2 + b, [a, b]),
        (x**4 + p*x**2 + q, [p, q]),
        (x**4 + p*x + q, [p, q]),
    ]
    for f, parameters in families:
        cases = real_root_classification(f, x)
        counts = [count for _, count in cases]
        assert counts == sorted(counts, key=lambda k: -1 if k is None else k) and len(set(counts)) == len(counts)
        seen: set[Optional[int]] = set()
        # integers and halves in a small box: many points on the discriminant and with a zero leading coefficient
        for _ in range(60):
            values: dict[Symbol, Expr] = {s: Rational(rnd.randint(-6, 6), rnd.choice([1, 1, 2, 3])) for s in parameters}
            numeric = f
            for symbol, value in values.items():
                numeric = numeric.subs(symbol, value)
            expected = _distinct_real_roots(sympy.expand(sympy_expr(to_sympy(numeric))))
            assert _evaluate(cases, values) == [expected], (f, values)
            seen.add(expected)
        assert len(seen) >= 2


def test_boundaries_and_known() -> None:
    # the quadratic: the discriminant
    assert str(real_root_classification(x**2 + b*x + c, x)) == '[(b**2 - 4*c < 0, 0), (Eq(b**2 - 4*c, 0), 1), (b**2 - 4*c > 0, 2)]'
    cubic = real_root_classification(x**3 + p*x + q, x)
    # on the discriminant curve: (p, q) = (-3, 2) is (x - 1)**2 (x + 2); the origin is the triple root
    assert _evaluate(cubic, {p: Integer(-3), q: Integer(2)}) == [2] and _evaluate(cubic, {p: Integer(0), q: Integer(0)}) == [1]
    assert _evaluate(cubic, {p: Integer(-3), q: Integer(0)}) == [3] and _evaluate(cubic, {p: Integer(1), q: Integer(5)}) == [1]
    # x**4 + p x**2 + q at p = -2, q = 1: (x**2 - 1)**2, two double roots; at q = 0, p < 0: three roots
    quartic = real_root_classification(x**4 + p*x**2 + q, x)
    assert _evaluate(quartic, {p: Integer(-2), q: Integer(1)}) == [2] and _evaluate(quartic, {p: Integer(-1), q: Integer(0)}) == [3]
    assert _evaluate(quartic, {p: Integer(0), q: Integer(0)}) == [1] and _evaluate(quartic, {p: Integer(1), q: Integer(1)}) == [0]
    # the degree drops: a x + b
    linear = real_root_classification(a*x + b, x)
    assert _evaluate(linear, {a: Integer(0), b: Integer(0)}) == [None] and _evaluate(linear, {a: Integer(0), b: Integer(1)}) == [0]
    assert _evaluate(linear, {a: Integer(2), b: Integer(1)}) == [1]
    assert discrimination_sequence(x**3 + p*x + q, x) == [3, -6*p, -4*p**3 - 27*q**2]
    assert discrimination_sequence(Integer(5), x) == [] and real_root_classification(Integer(0), x) == [(true, None)]
    with pytest.raises(NotImplementedError):
        real_root_classification(x**7 + a, x)
    with pytest.raises(ValueError):
        real_root_classification(1/x + a, x)
