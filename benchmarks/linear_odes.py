"""Run the linear ODE solvers of sympy-extras (Kovacic's algorithm, the
polynomial/rational/hyperexponential solutions and reduction of order of
``dsolve_linear``) on the homogeneous linear equations of the Kamke
collection and verify every solution with ``checkodesol``.

Usage::

    python benchmarks/linear_odes.py [--collections kamke1,kamke2]
        [--start N] [--limit N] [--timeout SECONDS] [--output results.json]

For every equation which ``LinearOperator.from_equation`` accepts (a
homogeneous linear equation with rational coefficients) the outcome is
``verified`` (as many independent solutions as the order, all confirmed by
``checkodesol``), ``partial`` (fewer solutions, all confirmed),
``unverified`` (a solution which could not be checked), ``failed`` (no
solution) or ``timeout``.
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
from sympy.solvers.ode import checkodesol

from sympy_extras._timeout import attempt
from sympy_extras.settings import configure
from sympy_extras.solvers.linear_ode import LinearOperator, dsolve_linear

from kamke_data import ODEEntry, load, x, y


def run(entry: ODEEntry, timeout: float) -> tuple[str, float, int]:
    started = time.time()
    try:
        L = LinearOperator.from_equation(entry.equation, y(x))
    except ValueError:
        return 'not linear', 0.0, 0
    with configure(timeout=timeout):
        found = attempt(lambda: dsolve_linear(entry.equation, y(x), use_dsolve=False), 3*timeout)
    elapsed = time.time() - started
    if found is None:
        return 'timeout', elapsed, 0
    if not found:
        return 'failed', elapsed, 0
    for solution in found:
        verdict = attempt(lambda: checkodesol(entry.equation, Eq(y(x), solution), y(x)), timeout)
        if verdict is None:
            return 'unverified', time.time() - started, len(found)
        if not verdict[0]:
            return 'wrong', time.time() - started, len(found)
    return ('verified' if len(found) >= L.order else 'partial'), time.time() - started, len(found)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collections', default='kamke1,kamke2')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('--output', default='')
    args = parser.parse_args(argv)
    entries = load(tuple(args.collections.split(',')))
    if not entries:
        print("no equations could be loaded (network unavailable?)")
        return 1
    entries = entries[args.start:]
    counts: dict[str, int] = {}
    results: list[dict[str, object]] = []
    seen = 0
    for entry in entries:
        status, elapsed, number = run(entry, args.timeout)
        if status == 'not linear':
            continue
        seen += 1
        counts[status] = counts.get(status, 0) + 1
        results.append({'name': entry.name, 'order': entry.order, 'equation': str(entry.equation),
                        'status': status, 'solutions': number, 'time': round(elapsed, 2)})
        print("%-10s order %d  %-10s %d solutions  %5.1fs  %s" % (entry.name, entry.order, status, number,
              elapsed, entry.equation), flush=True)
        if args.limit and seen >= args.limit:
            break
    print()
    print("linear equations:", seen)
    print(dict(sorted(counts.items())))
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps({'counts': counts, 'results': results}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
