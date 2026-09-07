"""Draw random equations from the collections and verify the solutions
independently.

Usage::

    python benchmarks/verify_random.py [--sample N] [--seed N] [--timeout S]
        [--collections kamke1,kamke2]

For every sampled equation of the Kamke collection ``solve_ode`` (SymPy's
``dsolve`` first, the symmetry method when it fails) is run, and every
solution is verified twice: symbolically with ``checkodesol`` and
numerically, by evaluating the residual of the equation at random points
with random values of the constants and parameters (an implicit solution
`G(x, y) = 0` is differentiated implicitly and its points are found with
``nsolve``). A solution which fails the numerical check is reported as
WRONG; one which passes neither check is UNVERIFIED.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time
from typing import Optional, Union

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sympy import Basic, Derivative, Eq, Expr, Symbol, Rational, Integral, nsolve, diff, N
from sympy.solvers.ode import checkodesol

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr
from sympy_extras.solvers import solve_ode

from kamke_data import ODEEntry, load, x, y

PRECISION = 30
TOLERANCE = Rational(1, 10)**12

#: values for the symbols of an equation
Point = dict[Union[Basic, complex], Union[Basic, complex]]


def _residual_explicit(equation: Expr, value: Expr, point: Point) -> Optional[Expr]:
    """The residual of the equation at an explicit solution ``y = value``."""
    e = equation.subs(y(x), value).doit()
    r = e.subs(point)
    r = N(r, PRECISION)
    return r if isinstance(r, Expr) and r.is_number else None


def _residual_implicit(equation: Expr, G: Expr, order: int, point: Point) -> Optional[Expr]:
    """The residual at a point of the curve ``G(x, y) = 0``: the derivatives
    of ``y`` come from implicit differentiation."""
    ysym = Symbol('y_value')
    g = G.subs(y(x), ysym)
    numeric = g.subs(point)
    x0 = point[x]
    y0: Optional[Basic] = None
    for guess in (Rational(1, 2), 1, -1, 2, Rational(-3, 2)):
        try:
            candidate = nsolve(numeric.subs(x, x0), ysym, guess, prec=PRECISION)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        if abs(N(numeric.subs({x: x0, ysym: candidate}), PRECISION)) < TOLERANCE:
            y0 = candidate
            break
    if y0 is None:
        return None
    derivatives: list[Expr] = [ysym]
    y1 = -diff(g, x)/diff(g, ysym)
    derivatives.append(y1)
    for _ in range(order - 1):
        last = derivatives[-1]
        derivatives.append(diff(last, x) + diff(last, ysym)*y1)
    e = equation
    for d in sorted(e.atoms(Derivative), key=lambda d: -sum(int(n) for _, n in d.variable_count)):
        if d.expr == y(x):
            k = sum(int(n) for _, n in d.variable_count)
            e = e.subs(d, derivatives[k])
    e = e.subs(y(x), ysym)
    r = N(e.subs(point).subs({x: x0, ysym: y0}), PRECISION)
    return r if isinstance(r, Expr) and r.is_number else None


def numeric_check(entry: ODEEntry, solution: Eq, rng: random.Random, trials: int = 3) -> Optional[bool]:
    """``True`` if the residual vanishes at random points, ``False`` if it
    does not, ``None`` if it could not be evaluated."""
    if solution.has(Integral):
        return None
    parameters = sorted((solution.free_symbols | entry.equation.free_symbols) - {x}, key=str)
    outcomes: list[bool] = []
    for _ in range(trials):
        point: Point = {p: Rational(rng.randint(1, 9), rng.randint(1, 4)) for p in parameters}
        point[x] = Rational(rng.randint(1, 12), rng.randint(1, 5))
        if solution.lhs == y(x):
            r = _residual_explicit(entry.equation, as_expr(solution.rhs), point)
        else:
            r = _residual_implicit(entry.equation, as_expr(solution.lhs - solution.rhs), entry.order, point)
        if r is None or not r.is_finite:
            continue
        outcomes.append(bool(abs(r) < TOLERANCE))
    if not outcomes:
        return None
    return all(outcomes)


def verify(entry: ODEEntry, rng: random.Random, timeout: float) -> tuple[str, list[str]]:
    solutions = attempt(lambda: solve_ode(entry.equation, y(x), check=False, timeout=timeout), 4*timeout)
    if not solutions:
        return 'failed', []
    verdicts: list[str] = []
    for sol in solutions:
        symbolic = attempt(lambda: checkodesol(entry.equation, sol, y(x)), timeout)
        symbolic_ok = symbolic is not None and bool(symbolic[0])
        numeric = attempt(lambda: numeric_check(entry, sol, rng), timeout)
        if numeric is False:
            verdicts.append('WRONG: %s' % sol)
        elif numeric or symbolic_ok:
            verdicts.append('verified: %s' % sol)
        else:
            verdicts.append('unverified: %s' % sol)
    if any(v.startswith('WRONG') for v in verdicts):
        status = 'WRONG'
    elif any(v.startswith('verified') for v in verdicts):
        status = 'verified'
    else:
        status = 'unverified'
    return status, verdicts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('--collections', default='kamke1,kamke2')
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    entries = load(tuple(args.collections.split(',')))
    if not entries:
        print("no equations could be loaded")
        return 1
    sample = rng.sample(entries, min(args.sample, len(entries)))
    counts: dict[str, int] = {}
    for entry in sample:
        started = time.time()
        status, verdicts = verify(entry, rng, args.timeout)
        counts[status] = counts.get(status, 0) + 1
        print("%-10s order %d %-10s %5.1fs  %s" % (entry.name, entry.order, status, time.time() - started, entry.equation), flush=True)
        for v in verdicts:
            print("    ", v)
    print()
    print("sample:", len(sample), dict(sorted(counts.items())))
    return 1 if counts.get('WRONG') else 0


if __name__ == '__main__':
    sys.exit(main())
