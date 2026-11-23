r"""The printed form means the expression: seeded random trees
(unevaluated, with negative coefficients, nested powers and quotients,
functions, ``mod``, relations and connectives) are printed, the LaTeX is
read back by the small reader of ``latex_reader.py`` and evaluated at
random points, and the value is compared with the value of the tree. A
sample of what the printer writes is also compiled with pdflatex.
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Optional

from sympy_extras.core import (Add, And, Eq, Equivalent, Ge, Gt, Implies, Integer, Le, Lt, Mod, Mul, Ne, Not, Or,
    Pow, Rational, Symbol, Xor)
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean
from sympy_extras.core.relational import Relational
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.trigonometric import Cos, Sin
from sympy_extras.printing.latex import LatexPrinter, LatexSettings
from sympy_extras.printing.tests.corpus import DIFFERENT, SAME
from sympy_extras.printing.tests.latex_reader import ReadError, Value, read_latex

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
SYMBOLS: list[Symbol] = [x, y, z]
NUMBERS: list[Expr] = [Integer(2), Integer(3), Integer(-1), Integer(-2), Integer(1), Rational(1, 2), Rational(-2, 3),
                       Rational(3, 4)]
EXPONENTS: list[Expr] = [Integer(2), Integer(3), Integer(-1), Integer(-2)]
ROOTS: list[Expr] = [Rational(1, 2), Rational(-1, 2), Rational(1, 3), Rational(2, 3), Rational(-3, 2)]

SETTINGS: list[LatexSettings] = [
    LatexSettings(),
    LatexSettings(fold_short_frac=True),
    LatexSettings(fold_frac_powers=True, mul_symbol='dot'),
    LatexSettings(order='none', long_frac_ratio=1),
    LatexSettings(root_notation=False, mul_symbol='times'),
]


def random_expression(rng: random.Random, depth: int) -> Expr:
    """An unevaluated tree of sums, products, powers, roots of positive
    bases, functions and ``mod``."""
    if depth == 0 or rng.random() < 0.15:
        return rng.choice(SYMBOLS) if rng.random() < 0.6 else rng.choice(NUMBERS)
    kind = rng.randrange(12)
    sub = depth - 1
    if kind <= 2:
        return Add(*[random_expression(rng, sub) for _ in range(rng.randint(2, 3))])
    if kind <= 5:
        return Mul(*[random_expression(rng, sub) for _ in range(rng.randint(2, 3))])
    if kind == 6:
        return Pow(random_expression(rng, sub), rng.choice(EXPONENTS))
    if kind == 7:
        # a positive base: the root is real
        return Pow(Add(Abs(random_expression(rng, sub)), Integer(1)), rng.choice(ROOTS))
    if kind == 8:
        return Pow(Add(Abs(random_expression(rng, sub)), Integer(1)), random_expression(rng, 1))
    if kind == 9:
        return rng.choice([Sin, Cos])(random_expression(rng, sub))
    if kind == 10:
        return rng.choice([Exp(random_expression(rng, 1)), Log(Add(Abs(random_expression(rng, sub)), Integer(1))),
                           Abs(random_expression(rng, sub))])
    return Mod(random_expression(rng, sub), Add(Abs(random_expression(rng, 1)), Integer(2)))


def random_formula(rng: random.Random, depth: int) -> Boolean:
    """A tree of connectives over relations of random expressions."""
    if depth == 0 or rng.random() < 0.3:
        relation = rng.choice([Lt, Le, Gt, Ge, Eq, Ne])
        return relation(random_expression(rng, 2), random_expression(rng, 2))
    kind = rng.randrange(6)
    sub = depth - 1
    if kind == 0:
        return Not(random_formula(rng, sub))
    if kind == 1:
        return And(random_formula(rng, sub), random_formula(rng, sub))
    if kind == 2:
        return Or(random_formula(rng, sub), random_formula(rng, sub))
    if kind == 3:
        return Implies(random_formula(rng, sub), random_formula(rng, sub))
    if kind == 4:
        return Xor(random_formula(rng, sub), random_formula(rng, sub))
    return Equivalent(random_formula(rng, sub), random_formula(rng, sub))


def number(e: Basic, point: Mapping[Symbol, float]) -> float:
    """The value of an expression of the grammar, by the tree."""
    if isinstance(e, Symbol):
        return point[e]
    if isinstance(e, Rational):
        return e.p / e.q
    if isinstance(e, Add):
        return math.fsum(number(t, point) for t in e.args)
    if isinstance(e, Mul):
        return math.prod(number(t, point) for t in e.args)
    if isinstance(e, Pow):
        return float(number(e.base, point) ** number(e.exp, point))
    if isinstance(e, Mod):
        p, q = number(e.args[0], point), number(e.args[1], point)
        return p - q * math.floor(p / q)
    if isinstance(e, Sin):
        return math.sin(number(e.args[0], point))
    if isinstance(e, Cos):
        return math.cos(number(e.args[0], point))
    if isinstance(e, Exp):
        return math.exp(number(e.args[0], point))
    if isinstance(e, Log):
        return math.log(number(e.args[0], point))
    if isinstance(e, Abs):
        return abs(number(e.args[0], point))
    raise TypeError('outside the grammar: %s' % (e,))


def truth(e: Basic, point: Mapping[Symbol, float]) -> bool:
    """The truth value of a formula of the grammar, by the tree."""
    if isinstance(e, Relational):
        p, q = number(e.lhs, point), number(e.rhs, point)
        close = math.isclose(p, q, rel_tol=1e-9, abs_tol=1e-9)
        return {'==': close, '!=': not close, '<': p < q, '<=': p <= q, '>': p > q, '>=': p >= q}[e.rel_op]
    if isinstance(e, Not):
        return not truth(e.args[0], point)
    # every operand is evaluated, as the reader does: no shortcut hides an undefined one
    if isinstance(e, And):
        return all([truth(t, point) for t in e.args])
    if isinstance(e, Or):
        return any([truth(t, point) for t in e.args])
    if isinstance(e, Implies):
        premise, conclusion = truth(e.args[0], point), truth(e.args[1], point)
        return (not premise) or conclusion
    if isinstance(e, Xor):
        return sum(truth(t, point) for t in e.args) % 2 == 1
    if isinstance(e, Equivalent):
        return len({truth(t, point) for t in e.args}) == 1
    raise TypeError('outside the grammar: %s' % (e,))


def _points(rng: random.Random) -> list[dict[Symbol, float]]:
    return [{s: rng.uniform(0.3, 2.7) * rng.choice([1, 1, -1]) for s in SYMBOLS} for _ in range(3)]


def _agree(expected: Value, got: Value) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected is got
    return math.isclose(expected, got, rel_tol=1e-7, abs_tol=1e-7)


def _defined(e: Basic, point: Mapping[Symbol, float]) -> Optional[Value]:
    """The value of the tree, ``None`` where it is not a moderate real
    number (a division by zero, an overflow, a value near a jump)."""
    try:
        if isinstance(e, Boolean):
            return truth(e, point)
        value = number(e, point)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return None
    if not math.isfinite(value) or abs(value) > 1e12:
        return None
    return value


def _stable(e: Basic, point: Mapping[Symbol, float]) -> bool:
    """Whether the value does not jump next to the point (``mod`` and the
    relations are discontinuous: a rounding error must not decide)."""
    values = [_defined(e, {s: v * (1 + shift) for s, v in point.items()}) for shift in (-1e-9, 0.0, 1e-9)]
    if any(v is None for v in values):
        return False
    first = values[0]
    return first is not None and all(v is not None and _agree(first, v) for v in values[1:])


def _check(trees: list[Basic], rng: random.Random) -> int:
    checked = 0
    for settings in SETTINGS:
        printer = LatexPrinter(settings)
        for tree in trees:
            tex = printer.doprint(tree)
            for point in _points(rng):
                expected = _defined(tree, point)
                if expected is None or not _stable(tree, point):
                    continue
                try:
                    got = read_latex(tex, {str(s): v for s, v in point.items()})
                except (ReadError, ZeroDivisionError, OverflowError, ValueError) as error:
                    raise AssertionError('%s printed as %s cannot be read: %s' % (tree, tex, error))
                assert _agree(expected, got), (str(tree), tex, settings, point, expected, got)
                checked += 1
    return checked


def test_random_expressions_read_back_with_their_value() -> None:
    rng = random.Random(30844)
    trees: list[Basic] = [random_expression(rng, rng.randint(2, 4)) for _ in range(1500)]
    assert _check(trees, rng) > 10000


def test_random_formulas_read_back_with_their_value() -> None:
    rng = random.Random(308440)
    trees: list[Basic] = [random_formula(rng, rng.randint(1, 3)) for _ in range(600)]
    assert _check(trees, rng) > 4000


def test_structurally_different_trees_print_differently() -> None:
    # the pairs of sympy/sympy#30844 which SymPy prints identically, and their kin
    a, b, c = x, y, z
    pairs: list[tuple[Basic, Basic]] = [
        (Mul(-1, Pow(a, 2)), Pow(Mul(-1, a), 2)),
        (Mul(Mul(a, Pow(b, -1)), Pow(c, -1)), Mul(a, Pow(Mul(b, Pow(c, -1)), -1))),
        (Mod(Mul(-1, a), b), Mul(-1, Mod(a, b))),
        (Mod(a, Mod(b, c)), Mod(Mod(a, b), c)),
        (Pow(Pow(a, b), c), Pow(a, Pow(b, c))),
        (Mul(2, Add(a, b)), Add(Mul(2, a), b)),
        (Not(And(a < b, b < c)), And(Not(a < b), b < c)),
        (Add(a, Mul(-1, Add(b, c))), Add(a, Mul(-1, b), c)),
        (Pow(Rational(-1, 2), a), Mul(-1, Pow(Rational(1, 2), a))),
        (Pow(Pow(Sin(a), 3), -2), Pow(Sin(a), -3)),
        (Pow(Pow(Sin(a), 3), 2), Pow(Sin(a), 3)),
    ]
    for settings in SETTINGS:
        printer = LatexPrinter(settings)
        for first, second in pairs:
            assert printer.doprint(first) != printer.doprint(second), (str(first), str(second), settings)


_PREAMBLE = r'''\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
'''


def test_the_output_compiles_with_pdflatex(tmp_path: Path) -> None:
    pdflatex = shutil.which('pdflatex')
    if pdflatex is None:
        return  # nothing to compile with
    rng = random.Random(3084)
    trees: list[Basic] = list(SAME) + [e for e, _, _ in DIFFERENT]
    trees += [random_expression(rng, 3) for _ in range(150)]
    trees += [random_formula(rng, 2) for _ in range(60)]
    formulas: list[str] = []
    for settings in (LatexSettings(), LatexSettings(fold_short_frac=True, fold_frac_powers=True,
                                                    inv_trig_style='power', decimal_separator='comma'),
                     LatexSettings(fold_func_brackets=True, mul_symbol='times', parenthesize_super=False)):
        printer = LatexPrinter(settings)
        formulas += [printer.doprint(tree) for tree in trees]
    source = tmp_path / 'formulas.tex'
    source.write_text(_PREAMBLE + '\n'.join('\\[ %s \\]' % tex for tex in formulas) + '\n\\end{document}\n')
    done = subprocess.run([pdflatex, '-interaction=nonstopmode', '-halt-on-error', source.name], cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)
    errors = [line for line in done.stdout.splitlines() if line.startswith('!')]
    assert done.returncode == 0 and not errors, (errors[:5], done.stdout[-1500:])
    assert len(formulas) > 1500
