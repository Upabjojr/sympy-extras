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

from sympy import Poly, Symbol, S, FiniteSet, Rational, And, Basic

from sympy_extras.assumptions import solve

x = Symbol('x')


def random_polynomial(rng: random.Random) -> Poly:
    degree = rng.randint(1, 5)
    coefficients = [rng.randint(-6, 6) for _ in range(degree + 1)]
    if coefficients[0] == 0:
        coefficients[0] = 1
    return Poly(coefficients, x)


def random_assumption(rng: random.Random) -> tuple[Basic, tuple[Rational, Rational, bool, bool]]:
    """An assumption on ``x`` and the interval (lower, upper, lower open,
    upper open) it describes."""
    kind = rng.randint(0, 3)
    if kind == 0:
        return x > 0, (S.Zero, S.Infinity, True, True)
    if kind == 1:
        return x < 0, (S.NegativeInfinity, S.Zero, True, True)
    lo = Rational(rng.randint(-8, 4), rng.randint(1, 3))
    hi = lo + Rational(rng.randint(1, 8), rng.randint(1, 3))
    if kind == 2:
        return And(x > lo, x < hi), (lo, hi, True, True)
    return And(x >= lo, x <= hi), (lo, hi, False, False)


def oracle(p: Poly, interval: tuple[Rational, Rational, bool, bool]) -> set[Basic]:
    lo, hi, lo_open, hi_open = interval
    roots: set[Basic] = set()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    mismatches = 0
    for _ in range(args.cases):
        p = random_polynomial(rng)
        assumption, interval = random_assumption(rng)
        expected = oracle(p, interval)
        result = solve(p.as_expr(), x, assumption)
        if not isinstance(result, FiniteSet) and result is not S.EmptySet:
            mismatches += 1
            print("unexpected result", p.as_expr(), assumption, result)
            continue
        found = set(result.args) if isinstance(result, FiniteSet) else set()
        expected_values = sorted(v.evalf(30) for v in expected)
        found_values = sorted(v.evalf(30) for v in found)
        if len(expected_values) != len(found_values) or any(abs(a - b) > Rational(1, 10)**20 for a, b in zip(expected_values, found_values)):
            mismatches += 1
            print("MISMATCH", p.as_expr(), assumption, "expected", expected, "got", result)
    print("cases:", args.cases, "mismatches:", mismatches)
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main())
