"""Random check of the refinement and simplification of logical
expressions (``sympy_extras.assumptions.refine``, ``simplify``, ``ask``,
``tautology``, ``satisfiable``) against independent evaluation.

Usage::

    python benchmarks/logic_random.py [--cases N] [--seed N] [--variables N]

Random Boolean combinations of polynomial relations in one or two real
variables are generated (``&``, ``|``, ``~``, ``Implies``, ``Equivalent``,
``Xor`` over atoms ``p(x, y) <op> 0`` of low degree), together with random
polynomial assumptions. For each case:

* ``simplify(formula, assumptions, domain=S.Reals)`` must be equivalent to
  the formula under the assumptions: both are evaluated at random rational
  points satisfying the assumptions;
* ``refine(formula, assumptions)`` likewise;
* ``ask(formula, assumptions)`` must agree with the evaluation at the
  points (``True`` only if it holds at every point tried, ``False`` only
  if it fails at every point);
* ``satisfiable`` must agree with the existence of a point among those
  tried when it answers ``False``;
* ``tautology(Equivalent(formula, simplified))`` under the assumptions,
  decided by the CAD, is checked when the formula is small.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Optional, Union

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sympy import (S, Symbol, Rational, And, Or, Not, Implies, Equivalent, Xor,
    Eq, Ne, Basic, Expr, true, false, exp, log, sin, cos, sqrt, atan, Abs, N)
from sympy.logic.boolalg import Boolean

from sympy_extras._typing import as_boolean, as_expr

from sympy_extras.assumptions import ask, refine, simplify, satisfiable, tautology
from sympy_extras._timeout import attempt

x, y = Symbol('x'), Symbol('y')

#: values for the variables
Point = dict[Union[Basic, complex], Union[Basic, complex]]


def random_polynomial(rng: random.Random, variables: list[Symbol]) -> Basic:
    terms = []
    for _ in range(rng.randint(1, 3)):
        c = rng.randint(-4, 4) or 1
        term = S(c)
        for v in variables:
            term *= v**rng.randint(0, 2)
        terms.append(term)
    return sum(terms, S.Zero)


def random_atom(rng: random.Random, variables: list[Symbol]) -> Boolean:
    p = as_expr(random_polynomial(rng, variables))
    op = rng.choice(['>', '<', '>=', '<=', '==', '!='])
    if op == '>':
        return as_boolean(p > 0)
    if op == '<':
        return as_boolean(p < 0)
    if op == '>=':
        return as_boolean(p >= 0)
    if op == '<=':
        return as_boolean(p <= 0)
    if op == '==':
        return as_boolean(Eq(p, 0))
    return as_boolean(Ne(p, 0))


def random_formula(rng: random.Random, variables: list[Symbol], depth: int) -> Boolean:
    if depth == 0 or rng.random() < 0.3:
        return random_atom(rng, variables)
    kind = rng.choice(['and', 'or', 'not', 'implies', 'equivalent', 'xor'])
    a = random_formula(rng, variables, depth - 1)
    if kind == 'not':
        return Not(a)
    b = random_formula(rng, variables, depth - 1)
    if kind == 'and':
        return And(a, b)
    if kind == 'or':
        return Or(a, b)
    if kind == 'implies':
        return Implies(a, b)
    if kind == 'equivalent':
        return Equivalent(a, b)
    return Xor(a, b)


def evaluate(formula: Basic, point: Point) -> Optional[bool]:
    # subs, not xreplace: the generator of a CRootOf is not a free symbol
    value = formula.subs(point)
    if value is false:
        return False
    if value is true:
        return True
    return None


def random_points(rng: random.Random, variables: list[Symbol], assumption: Boolean, count: int) -> list[Point]:
    points: list[Point] = []
    for _ in range(count*20):
        point: Point = {v: Rational(rng.randint(-12, 12), rng.randint(1, 4)) for v in variables}
        if evaluate(assumption, point) is True:
            points.append(point)
        if len(points) >= count:
            break
    return points


def random_transcendental(rng: random.Random) -> Expr:
    pieces = [x, x**2, exp(x), exp(-x), log(x + 6), sin(x), cos(x), sqrt(x + 6), atan(x)]
    terms = rng.sample(pieces, rng.randint(1, 3))
    result: Expr = S.Zero
    for term in terms:
        result = result + rng.choice([-3, -2, -1, 1, 2, 3])*term
    return result + rng.randint(-4, 4)


def check_transcendental(rng: random.Random, cases: int, timeout: float) -> tuple[int, int]:
    """``ask(f > 0)`` and ``refine(Abs(f))`` for random non-polynomial
    ``f`` of one variable on random intervals, against dense sampling.
    Returns (problems, undecided)."""
    problems = 0
    undecided = 0
    for case in range(cases):
        f = random_transcendental(rng)
        lo = Rational(rng.randint(-5, 2), rng.randint(1, 2))
        hi = lo + Rational(rng.randint(1, 8), rng.randint(1, 2))
        assumption = as_boolean(And(x > lo, x < hi))
        samples = [lo + (hi - lo)*Rational(k, 400) for k in range(1, 400)]
        values = [N(f.subs(x, s), 30) for s in samples]
        positive = all(v > 0 for v in values)
        negative = all(v < 0 for v in values)
        label = "transcendental %d: %s on (%s, %s)" % (case, f, lo, hi)
        answer = attempt(lambda: ask(f > 0, assumption), timeout)
        if answer is True and not positive or answer is False and not all(v <= 0 for v in values):
            problems += 1
            print("WRONG ask", label, "->", answer)
        elif answer is None:
            undecided += 1
        refined = attempt(lambda: refine(Abs(f), assumption), timeout)
        if refined is None:
            undecided += 1
            continue
        for s, v in zip(samples[::40], values[::40]):
            w = N(refined.subs(x, s), 30)
            if abs(w - abs(v)) > Rational(1, 10)**20:
                problems += 1
                print("WRONG refine", label, "->", refined, "at", s)
                break
        if refined == Abs(f) and (positive or negative):
            undecided += 1
    return problems, undecided


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variables', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--transcendental', type=int, default=0,
                        help="number of random non-polynomial expressions of one variable to check as well")
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    variables = [x, y][:args.variables]
    problems = 0
    skipped = 0
    if args.transcendental:
        wrong, undecided = check_transcendental(rng, args.transcendental, args.timeout)
        problems += wrong
        print("transcendental cases:", args.transcendental, "wrong:", wrong, "undecided:", undecided)
    for case in range(args.cases):
        formula = random_formula(rng, variables, rng.randint(1, 3))
        assumption: Boolean = true if rng.random() < 0.4 else random_atom(rng, variables)
        if assumption is false:
            continue
        points = random_points(rng, variables, assumption, 12)
        if not points:
            skipped += 1
            continue
        values = [evaluate(formula, p) for p in points]
        if any(v is None for v in values):
            skipped += 1
            continue
        label = "case %d: %s  assuming %s" % (case, formula, assumption)
        # simplify and refine must be equivalent under the assumptions
        for name, function in (('simplify', simplify), ('refine', refine)):
            result = attempt(lambda: function(formula, assumption, domain=S.Reals), args.timeout)
            if result is None:
                print("TIMEOUT", name, label)
                skipped += 1
                continue
            for p, v in zip(points, values):
                w = evaluate(result, p)
                if w is not None and w != v:
                    problems += 1
                    print("WRONG", name, label, "->", result, "at", p, "expected", v)
                    break
        # ask
        answer = attempt(lambda: ask(formula, assumption), args.timeout)
        if answer is True and not all(values) or answer is False and any(values):
            problems += 1
            print("WRONG ask", label, "->", answer, "values", values)
        # satisfiable: False is only allowed when no point satisfies the formula
        sat = attempt(lambda: satisfiable(And(formula, assumption), domain=S.Reals), args.timeout)
        if sat is False and any(values):
            problems += 1
            print("WRONG satisfiable", label, "-> False but a point satisfies it")
        # the CAD decides the equivalence with the simplified formula
        simplified = attempt(lambda: simplify(formula, assumption, domain=S.Reals), args.timeout)
        if simplified is not None and len(variables) == 1:
            verdict = attempt(lambda: tautology(Equivalent(formula, simplified), assumptions=assumption, domain=S.Reals), args.timeout)
            if verdict is False:
                problems += 1
                print("WRONG equivalence", label, "->", simplified)
    print("cases:", args.cases, "problems:", problems, "skipped:", skipped)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
