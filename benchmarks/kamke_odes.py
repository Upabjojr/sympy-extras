"""Run SymPy's ``dsolve`` and the Lie symmetry solver of sympy-extras on
the Kamke collection of ordinary differential equations and verify every
solution with ``checkodesol``.

After ``dsolve`` the linear solvers of sympy-extras (Kovacic's algorithm,
rational and hyperexponential solutions, Bessel/Whittaker/hypergeometric
recognition) and the first order Abel/Chini/Lagrange solvers are tried
("extras"), then the Lie symmetry solver.

Usage::

    python benchmarks/kamke_odes.py [--collections kamke1,kamke2]
        [--start N] [--limit N] [--timeout SECONDS] [--output results.json]

The equations are downloaded from Maxima's repository on first use (see
``kamke_data.py``). For every equation the outcome of ``dsolve`` and, when
it fails, of ``dsolve_lie`` is recorded as one of ``verified`` (a solution
which ``checkodesol`` confirms), ``unverified`` (a solution which could not
be checked in time), ``failed`` (no solution) or ``timeout``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sympy.core.relational import Eq
from sympy.solvers.ode import dsolve, checkodesol

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import Rational

from sympy_extras._timeout import attempt
from sympy_extras.settings import configure
from sympy_extras.solvers import dsolve_lie, dsolve_linear, dsolve_first_order

from kamke_data import ODEEntry, load, x, y


def _check(entry: ODEEntry, solutions: list[Eq], timeout: float) -> str:
    for sol in solutions:
        verdict = attempt(lambda: checkodesol(entry.equation, sol, y(x)), timeout)
        if verdict is not None and verdict[0]:
            return 'verified'
    return 'unverified'


def run_sympy(entry: ODEEntry, timeout: float) -> tuple[str, float]:
    started = time.time()
    result = attempt(lambda: dsolve(entry.equation, y(x)), timeout)
    elapsed = time.time() - started
    if result is None:
        return ('timeout' if elapsed >= timeout*0.95 else 'failed'), elapsed
    solutions = [s for s in (result if isinstance(result, list) else [result]) if isinstance(s, Eq)]
    if not solutions:
        return 'failed', elapsed
    return _check(entry, solutions, timeout), time.time() - started


def run_lie(entry: ODEEntry, timeout: float) -> tuple[str, float]:
    started = time.time()
    result = attempt(lambda: dsolve_lie(entry.equation, y(x), check=False, timeout=timeout), 4*timeout)
    elapsed = time.time() - started
    if result is None:
        return 'timeout', elapsed
    if not result:
        return 'failed', elapsed
    return _check(entry, result, timeout), time.time() - started


def _numeric_residual(entry: ODEEntry, solution: Expr) -> Optional[float]:
    """The largest residual of an explicit solution at a few points, for
    solutions with special functions which ``checkodesol`` cannot
    simplify."""
    from sympy import N
    lhs = entry.equation.lhs - entry.equation.rhs if isinstance(entry.equation, Eq) else entry.equation
    residual = lhs.subs(y(x), solution).doit()
    values = {s: Rational(3, 10) + Rational(k, 7) for k, s in enumerate(sorted(residual.free_symbols - {x}, key=str))}
    worst = 0.0
    for point in (Rational(7, 10), Rational(13, 10), Rational(21, 10)):
        value = N(residual.subs(values).subs(x, point), 20)
        if not value.is_number or not value.is_finite:
            return None
        worst = max(worst, float(abs(value)))
    return worst


def run_extras(entry: ODEEntry, timeout: float) -> tuple[str, float]:
    """The linear (Kovacic, rational/hyperexponential, special functions)
    and first order (Abel, Chini, Lagrange) solvers of sympy-extras."""
    started = time.time()
    solutions: list[Basic] = []
    with configure(timeout=timeout):
        try:
            linear = attempt(lambda: dsolve_linear(entry.equation, y(x)), 4*timeout)
        except ValueError:
            linear = None
        if linear:
            solutions = [Eq(y(x), s) for s in linear]
            if len(linear) < entry.order:
                return 'partial', time.time() - started
        else:
            first = attempt(lambda: dsolve_first_order(entry.equation, y(x)), 4*timeout) if entry.order == 1 else None
            if isinstance(first, Eq):
                solutions = [first]
            elif first is not None:
                solutions = [s for s in first.args if isinstance(s, Eq)]
    elapsed = time.time() - started
    if not solutions:
        return ('timeout' if elapsed >= 4*timeout*0.95 else 'failed'), elapsed
    status = _check(entry, [s for s in solutions if isinstance(s, Eq)], timeout)
    if status != 'verified' and all(isinstance(s, Eq) and s.lhs == y(x) for s in solutions):
        residuals = [_numeric_residual(entry, s.rhs) for s in solutions if isinstance(s, Eq)]
        if residuals and all(r is not None and r < 1e-8 for r in residuals):
            status = 'verified numerically'
    return status, time.time() - started


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collections', default='kamke1,kamke2')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('--output', default='')
    parser.add_argument('--lie-only-on-failure', action='store_true', default=True)
    args = parser.parse_args(argv)
    entries = load(tuple(args.collections.split(',')))
    if not entries:
        print("no equations could be loaded (network unavailable?)")
        return 1
    entries = entries[args.start:]
    if args.limit:
        entries = entries[:args.limit]
    results: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {'sympy': {}, 'extras': {}, 'lie': {}, 'combined': {}}
    for entry in entries:
        status, elapsed = run_sympy(entry, args.timeout)
        record: dict[str, object] = {'name': entry.name, 'order': entry.order,
                                     'equation': str(entry.equation), 'sympy': status,
                                     'sympy_time': round(elapsed, 2)}
        lie_status = ''
        extras_status = ''
        if status != 'verified':
            extras_status, extras_elapsed = run_extras(entry, args.timeout)
            record['extras'] = extras_status
            record['extras_time'] = round(extras_elapsed, 2)
            counts['extras'][extras_status] = counts['extras'].get(extras_status, 0) + 1
        if status != 'verified' and not extras_status.startswith('verified'):
            lie_status, lie_elapsed = run_lie(entry, args.timeout)
            record['lie'] = lie_status
            record['lie_time'] = round(lie_elapsed, 2)
            counts['lie'][lie_status] = counts['lie'].get(lie_status, 0) + 1
        combined = 'verified' if 'verified' in (status, lie_status) else (
            extras_status if extras_status.startswith('verified') else status)
        record['combined'] = combined
        counts['sympy'][status] = counts['sympy'].get(status, 0) + 1
        counts['combined'][combined] = counts['combined'].get(combined, 0) + 1
        results.append(record)
        print("%-10s order %d  sympy=%-10s extras=%-21s lie=%-10s  %s" % (entry.name, entry.order, status,
              extras_status or '-', lie_status or '-', entry.equation), flush=True)
    print()
    print("equations:", len(results))
    for key, table in counts.items():
        print("%-9s" % key, dict(sorted(table.items())))
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps({'counts': counts, 'results': results}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
