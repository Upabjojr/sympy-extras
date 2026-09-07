"""Random check of ``sympy_extras.assumptions.solve`` on polynomial
equations with sign assumptions against an independent oracle.

Usage::

    python benchmarks/solve_random.py [--cases N] [--seed N]

For a random polynomial ``p(x)`` with integer coefficients and random
assumptions on ``x`` (an interval given by inequalities, or a sign), the
solutions returned by ``solve`` are compared with the real roots isolated
by SymPy's ``Poly.real_roots`` filtered by evaluating the assumptions at
high precision.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sympy import (Poly, Symbol, S, FiniteSet, Rational, And, Expr, exp, log,
    sin, cos, sqrt, lambdify, nsolve, N, sympify)
from sympy.logic.boolalg import Boolean

from sympy_extras._typing import as_boolean, as_expr

from sympy_extras.assumptions import solve

x = Symbol('x')


def random_polynomial(rng: random.Random) -> Poly:
    degree = rng.randint(1, 5)
    coefficients = [rng.randint(-6, 6) for _ in range(degree + 1)]
    if coefficients[0] == 0:
        coefficients[0] = 1
    return Poly(coefficients, x)


#: lower bound, upper bound, lower open, upper open
Bounds = tuple[Expr, Expr, bool, bool]


def random_assumption(rng: random.Random) -> tuple[Boolean, Bounds]:
    """An assumption on ``x`` and the interval (lower, upper, lower open,
    upper open) it describes."""
    kind = rng.randint(0, 3)
    if kind == 0:
        return as_boolean(x > 0), (S.Zero, S.Infinity, True, True)
    if kind == 1:
        return as_boolean(x < 0), (S.NegativeInfinity, S.Zero, True, True)
    lo = Rational(rng.randint(-8, 4), rng.randint(1, 3))
    hi = lo + Rational(rng.randint(1, 8), rng.randint(1, 3))
    if kind == 2:
        return as_boolean(And(x > lo, x < hi)), (lo, hi, True, True)
    return as_boolean(And(x >= lo, x <= hi)), (lo, hi, False, False)


def oracle(p: Poly, interval: Bounds) -> set[Expr]:
    lo, hi, lo_open, hi_open = interval
    roots: set[Expr] = set()
    for r in p.real_roots():
        value = r.evalf(60)
        if lo_open and not (value > lo):
            continue
        if not lo_open and not (value >= lo):
            continue
        if hi_open and not (value < hi):
            continue
        if not hi_open and not (value <= hi):
            continue
        roots.add(r)
    return roots


def random_transcendental(rng: random.Random) -> Expr:
    """A sum of a few terms among polynomials, exponentials, logarithms and
    sines, with a real root somewhere in ``(-5, 5)`` likely."""
    pieces = [x, x**2, exp(x), exp(-x), log(x + 6), sin(x), cos(x), sqrt(x + 6)]
    terms = rng.sample(pieces, rng.randint(1, 3))
    result: Expr = S.Zero
    for term in terms:
        result = result + rng.choice([-3, -2, -1, 1, 2, 3])*term
    return result + rng.randint(-4, 4)


def numeric_roots(f: Expr, lo: Expr, hi: Expr) -> list[Expr]:
    """Roots of ``f`` in ``(lo, hi)`` from sign changes on a fine grid,
    refined with ``nsolve``; roots of even multiplicity (touching minima
    of ``|f|``) are located by a golden section search in high
    precision."""
    import mpmath
    fn = lambdify(x, f, 'mpmath')
    # the interval is open: stay strictly inside it
    a = max(float(lo), -5.0) + 1e-9
    b = min(float(hi), 5.0) - 1e-9
    if a >= b:
        return []
    steps = 2000
    grid = [a + (b - a)*k/steps for k in range(steps + 1)]
    values = []
    for g in grid:
        try:
            values.append(float(fn(g)))
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            values.append(float('nan'))
    roots: list[Expr] = []

    def record(value: Expr) -> None:
        if abs(N(f.subs(x, value), 30)) > Rational(1, 10)**20:
            return
        if a < float(value) < b and all(abs(N(value - known, 30)) > Rational(1, 10)**8 for known in roots):
            roots.append(value)

    def refine_near(guess: float) -> None:
        try:
            r = nsolve(f, x, guess, prec=30)
        except (ValueError, TypeError, ZeroDivisionError):
            return
        record(as_expr(r))

    def refine_minimum(left_end: float, right_end: float) -> None:
        with mpmath.workdps(50):
            left, right = mpmath.mpf(left_end), mpmath.mpf(right_end)
            ratio = (mpmath.sqrt(5) - 1)/2
            for _ in range(300):
                c = right - ratio*(right - left)
                d = left + ratio*(right - left)
                if abs(fn(c)) < abs(fn(d)):
                    right = d
                else:
                    left = c
            point = (left + right)/2
            if abs(fn(point)) < mpmath.mpf(10)**-40:
                record(as_expr(sympify(point)))

    for k in range(steps):
        u, v = values[k], values[k + 1]
        if u != u or v != v:
            continue
        if u == 0:
            refine_near(grid[k])
        elif u*v < 0:
            refine_near((grid[k] + grid[k + 1])/2)
        elif 0 < k and abs(u) < 1e-4 and abs(u) <= abs(values[k - 1]) and abs(u) <= abs(v):
            refine_minimum(grid[k - 1], grid[k + 1])
    return roots


def check_transcendental(rng: random.Random, cases: int) -> int:
    mismatches = 0
    for _ in range(cases):
        f = random_transcendental(rng)
        lo = Rational(rng.randint(-5, 2), rng.randint(1, 2))
        hi = lo + Rational(rng.randint(1, 8), rng.randint(1, 2))
        assumption = as_boolean(And(x > lo, x < hi))
        expected = numeric_roots(f, lo, hi)
        result = solve(f, x, assumption)
        if not (isinstance(result, FiniteSet) or result is S.EmptySet):
            # ConditionSets and image sets are not compared, only reported
            print("undecided", f, assumption, "->", result)
            continue
        found = [as_expr(v) for v in result.args] if isinstance(result, FiniteSet) else []
        found_values = sorted(N(v, 30) for v in found)
        expected_values = sorted(N(v, 30) for v in expected)
        if len(found_values) != len(expected_values) or any(
                abs(a - b) > Rational(1, 10)**8 for a, b in zip(found_values, expected_values)):
            mismatches += 1
            print("MISMATCH", f, assumption, "expected", expected_values, "got", result)
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--transcendental', type=int, default=0,
                        help="number of random transcendental equations to check as well")
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    mismatches = 0
    if args.transcendental:
        mismatches += check_transcendental(rng, args.transcendental)
        print("transcendental cases:", args.transcendental, "mismatches so far:", mismatches)
    for _ in range(args.cases):
        p = random_polynomial(rng)
        assumption, interval = random_assumption(rng)
        expected = oracle(p, interval)
        result = solve(p.as_expr(), x, assumption)
        if not isinstance(result, FiniteSet) and result is not S.EmptySet:
            mismatches += 1
            print("unexpected result", p.as_expr(), assumption, result)
            continue
        found = {as_expr(v) for v in result.args} if isinstance(result, FiniteSet) else set()
        expected_values = sorted(v.evalf(30) for v in expected)
        found_values = sorted(v.evalf(30) for v in found)
        if len(expected_values) != len(found_values) or any(abs(a - b) > Rational(1, 10)**20 for a, b in zip(expected_values, found_values)):
            mismatches += 1
            print("MISMATCH", p.as_expr(), assumption, "expected", expected, "got", result)
    print("cases:", args.cases, "mismatches:", mismatches)
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main())
