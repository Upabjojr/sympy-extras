"""Stability conditions: numerical roots (SymPy) on random polynomials
and at random values of the parameters, and the classical conditions of
the textbooks."""
from __future__ import annotations

import random

import pytest
import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean, false, true
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.solvers.stability import hurwitz_determinants, hurwitz_matrix, is_stable, routh_array, stability_conditions

s, z, k, a, b, c = (Symbol(n) for n in 'szkabc')
ss = sympy.Symbol('s')


def _roots(p: sympy.Expr, x: sympy.Symbol) -> list[sympy.Expr]:
    # the squarefree part: nroots does not converge on multiple roots
    return list(sympy.Poly(p, x).sqf_part().nroots(n=20, maxsteps=200))


def _numerically_stable(p: sympy.Expr, x: sympy.Symbol, discrete: bool) -> bool:
    if discrete:
        return all(abs(root) < 1 for root in _roots(p, x))
    return all(sympy.re(root) < 0 for root in _roots(p, x))


def _holds(condition: Boolean, values: dict[Symbol, Expr]) -> bool:
    result = condition
    for symbol, value in values.items():
        substituted = result.subs(symbol, value)
        assert isinstance(substituted, Boolean)
        result = substituted
    assert result is true or result is false, result
    return result is true


def test_numeric_polynomials() -> None:
    rnd = random.Random(101)
    stable = unstable = 0
    for trial in range(60):
        n = rnd.randint(1, 5)
        if trial % 2:
            # a product of stable factors, sometimes spoiled
            p: Expr = Integer(1)
            for _ in range(n):
                p = p*(s + Rational(rnd.randint(1, 6), rnd.randint(1, 3)))
            if trial % 6 == 1:
                p = p*(s**2 - Rational(1, 3)*s + 2)
            p = p.expand()
        else:
            p = sum((Integer(rnd.randint(1, 9))*s**i for i in range(n + 1)), Integer(0))
        expected = _numerically_stable(sympy_expr(to_sympy(p)), ss, False)
        assert is_stable(p, s) == expected and is_stable(-p, s) == expected
        stable += expected
        unstable += not expected
        # the first column of the Routh array has no sign change exactly for the stable ones
        try:
            column = [row[0] for row in routh_array(p, s)]
        except ValueError:
            continue
        assert all(entry.is_positive for entry in column) == expected
    assert stable > 10 and unstable > 10


def test_discrete_numeric_polynomials() -> None:
    rnd = random.Random(102)
    sz = sympy.Symbol('z')
    stable = unstable = 0
    for _ in range(50):
        n = rnd.randint(1, 4)
        p: Expr = Integer(1)
        for _ in range(n):
            p = p*(z - Rational(rnd.randint(-5, 5), 4))
        if rnd.random() < 0.4:
            p = p*(z**2 - z + Rational(rnd.randint(1, 5), 4))
        p = p.expand()
        expected = _numerically_stable(sympy_expr(to_sympy(p)), sz, True)
        assert is_stable(p, z, discrete=True) == expected
        stable += expected
        unstable += not expected
    assert stable > 5 and unstable > 5
    # a root at 1 or on the circle is not stable
    assert not is_stable(z - 1, z, discrete=True) and not is_stable(z + 1, z, discrete=True)
    assert not is_stable(z**2 + 1, z, discrete=True) and is_stable(2*z - 1, z, discrete=True)


def test_classical_conditions() -> None:
    # the cubic a0 s**3 + a1 s**2 + a2 s + a3 with positive coefficients: a1 a2 > a0 a3
    assert hurwitz_determinants(s**3 + a*s**2 + b*s + c, s) == [a, a*b - c, a*b*c - c**2]
    assert hurwitz_matrix(s**2 + a*s + b, s) == ExprMatrix.from_rows([[a, 0], [1, b]])
    # Ogata, Modern Control Engineering: K/(s (s + 1) (s + 2)) in a unity feedback loop is stable for 0 < K < 6,
    # 1/(s + 1)**3 with the gain k for -1 < k < 8
    assert str(stability_conditions(s*(s + 1)*(s + 2) + k, s)) == '(k > 0) & (k < 6)'
    assert str(stability_conditions((s + 1)**3 + k, s)) == '(k > -1) & (k < 8)'
    # second order: both coefficients positive
    condition = stability_conditions(s**2 + a*s + b, s)
    for values, expected in [({a: 1, b: 1}, True), ({a: -1, b: 1}, False), ({a: 1, b: -1}, False), ({a: 0, b: 1}, False)]:
        assert _holds(condition, {key: Integer(value) for key, value in values.items()}) == expected
    assert stability_conditions(s**2 + 1, s) is false and stability_conditions(Integer(5), s) is true


def test_parameters_against_numerical_roots() -> None:
    rnd = random.Random(103)
    sz = sympy.Symbol('z')
    cases: list[tuple[Expr, Symbol, bool]] = [
        (s**3 + a*s**2 + b*s + 1, s, False),
        (s**4 + 2*s**3 + a*s**2 + b*s + 1, s, False),
        (a*s**2 + s + b, s, False),
        (z**2 + a*z + b, z, True),
        (z**3 + a*z + b, z, True),
        (s**2 + (a**2 - 1)*s + b/2, s, False),
    ]
    for p, variable, discrete in cases:
        condition = stability_conditions(p, variable, discrete)
        raw = stability_conditions(p, variable, discrete, simplify=False)
        agreed = {True: 0, False: 0}
        for _ in range(40):
            values: dict[Symbol, Expr] = {a: Rational(rnd.randint(-12, 12), 4), b: Rational(rnd.randint(-12, 12), 4)}
            numeric = sympy_expr(to_sympy(p.subs(a, values[a]).subs(b, values[b])))
            x = sz if discrete else ss
            if numeric == 0 or sympy.degree(numeric, x) < 1:
                continue
            roots = _roots(numeric, x)
            margin = min(abs(abs(r) - 1) for r in roots) if discrete else min(abs(sympy.re(r)) for r in roots)
            if margin < sympy.Float('1e-9') or sympy.degree(numeric, x) < sympy.degree(sympy_expr(to_sympy(p)), x):
                continue          # on the boundary, or the leading coefficient vanishes: a lower degree
            expected = _numerically_stable(numeric, x, discrete)
            assert _holds(condition, values) == expected and _holds(raw, values) == expected
            agreed[expected] += 1
        assert agreed[True] > 0 and agreed[False] > 5


def test_matrices() -> None:
    assert is_stable(ExprMatrix.from_rows([[-1, 2], [0, -3]])) and not is_stable(ExprMatrix.from_rows([[0, 1], [1, 0]]))
    assert is_stable(ExprMatrix.from_rows([[Rational(1, 2), 1], [0, Rational(-1, 3)]]), discrete=True)
    # the damped oscillator x'' + c x' + k x = 0
    condition = stability_conditions(ExprMatrix.from_rows([[0, 1], [-k, -c]]))
    assert _holds(condition, {k: Integer(1), c: Integer(1)}) and not _holds(condition, {k: Integer(1), c: Integer(-1)})
    assert not _holds(condition, {k: Integer(-1), c: Integer(1)})
    with pytest.raises(ValueError):
        is_stable(ExprMatrix.from_rows([[0, 1], [-k, -1]]))
    with pytest.raises(ValueError):
        stability_conditions(s**2 + 1)
    with pytest.raises(ValueError):
        stability_conditions(1/s + 1, s)
    with pytest.raises(ValueError):
        routh_array(s**3 + s, s)
