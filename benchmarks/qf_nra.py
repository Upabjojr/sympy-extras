"""Random check of ``satisfiable``, ``simplify`` and ``refine`` of
``sympy_extras.assumptions`` on SMT-LIB ``QF_NRA`` problems (the
Meti-Tarski family) against their recorded status.

Usage::

    python benchmarks/qf_nra.py [--sample N] [--seed N] [--timeout S]
        [--max-variables N]

For each sampled problem: ``satisfiable(formula, domain=S.Reals)`` is
compared with the ``:status`` of the file (a model returned is checked by
evaluation); ``simplify`` and ``refine`` of the formula over the reals
must give ``False`` only for unsatisfiable problems, ``True`` only for
satisfiable ones, and otherwise agree with the formula at random points.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sympy import S, Rational, Basic, true, false, count_ops
from sympy.logic.boolalg import Boolean

from sympy_extras.assumptions import satisfiable, simplify, refine
from sympy_extras._timeout import attempt

from smtlib_data import Problem, load, translate


def evaluate(formula: Boolean, point: dict[Basic, Basic]) -> Optional[bool]:
    value = formula.xreplace(point)
    if value is true:
        return True
    if value is false:
        return False
    return None


def check_rewrite(name: str, problem: Problem, result: Optional[Basic], rng: random.Random) -> str:
    if result is None:
        return 'timeout'
    if result is true:
        return 'ok' if problem.status == 'sat' else 'WRONG (True for an unsat problem)'
    if result is false:
        return 'ok' if problem.status == 'unsat' else 'WRONG (False for a sat problem)'
    if not isinstance(result, Boolean):
        return 'WRONG (not a Boolean: %s)' % (result,)
    for _ in range(20):
        point: dict[Basic, Basic] = {v: Rational(rng.randint(-20, 20), rng.randint(1, 5)) for v in problem.variables}
        a, b = evaluate(problem.formula, point), evaluate(result, point)
        if a is not None and b is not None and a != b:
            return 'WRONG (differs at %s)' % (point,)
    return 'ok'


def check_satisfiable(problem: Problem, result: object) -> str:
    if result is None:
        return 'undecided'
    if result is False:
        return 'ok' if problem.status == 'unsat' else 'WRONG (False for a sat problem)'
    if isinstance(result, dict):
        point: dict[Basic, Basic] = {k: v for k, v in result.items()}
        value = evaluate(problem.formula, point) if all(v in point for v in problem.variables) else None
        if value is False:
            return 'WRONG (the model does not satisfy the formula)'
        if problem.status == 'unsat':
            return 'WRONG (a model for an unsat problem)'
        return 'ok' if value is True else 'ok (model not checkable)'
    return 'ok'


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('--max-variables', type=int, default=2)
    parser.add_argument('--max-size', type=int, default=120)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    paths = load()
    if not paths:
        print("the benchmark files could not be fetched")
        return 1
    rng.shuffle(paths)
    counts: dict[str, dict[str, int]] = {'satisfiable': {}, 'simplify': {}, 'refine': {}}
    done = 0
    for path in paths:
        if done >= args.sample:
            break
        problem = translate(path.read_text(), path.stem)
        if problem is None or len(problem.variables) > args.max_variables or int(count_ops(problem.formula)) > args.max_size:
            continue
        done += 1
        started = time.time()
        sat = attempt(lambda: satisfiable(problem.formula, domain=S.Reals), args.timeout)
        verdicts = {'satisfiable': check_satisfiable(problem, sat)}
        simplified = attempt(lambda: simplify(problem.formula, domain=S.Reals), args.timeout)
        verdicts['simplify'] = check_rewrite('simplify', problem, simplified, rng)
        refined = attempt(lambda: refine(problem.formula, domain=S.Reals), args.timeout)
        verdicts['refine'] = check_rewrite('refine', problem, refined, rng)
        for key, verdict in verdicts.items():
            counts[key][verdict] = counts[key].get(verdict, 0) + 1
        print("%-40s %-6s vars=%d ops=%3d  sat=%-24s simplify=%-10s refine=%-10s %5.1fs" % (
            problem.name[:40], problem.status, len(problem.variables), int(count_ops(problem.formula)),
            verdicts['satisfiable'], verdicts['simplify'], verdicts['refine'], time.time() - started), flush=True)
    print()
    for key, table in counts.items():
        print("%-12s" % key, dict(sorted(table.items())))
    wrong = any('WRONG' in k for table in counts.values() for k in table)
    return 1 if wrong else 0


if __name__ == '__main__':
    sys.exit(main())
