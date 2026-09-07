"""Limits and series expansions under statement assumptions.

SymPy's :func:`~sympy.limit` and :func:`~sympy.series` know the
properties of a parameter only through the assumptions of its
:class:`~sympy.Symbol` (``Symbol('a', positive=True)``), and
``limit(exp(a*x), x, oo)`` with a plain ``a`` raises ``NotImplementedError:
Result depends on the sign of sign(a)``. Here the parameters carry the
statement assumptions (``a > 0``, ``a < 0``, ``element(a, S.Integers)``,
...) into SymPy through dummies with the matching core assumptions, and
when the result still depends on the sign of some expression the limit is
computed in each case (positive, zero, negative) compatible with the
assumptions and returned as a ``Piecewise``, the counterpart of
Mathematica's ``GenerateConditions``.
"""
from __future__ import annotations

import re
from typing import Optional, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.mul import Mul
from sympy.logic.boolalg import Boolean, And, true
from sympy.series.limits import Limit, limit as _limit
from sympy.sets.sets import Set

from sympy_extras._typing import as_expr, free_symbols

from .ask import Assumptions, _facts, _evaluate
from .facts import Facts, normalize
from .refine import _Refiner
from .sat import satisfiable

__all__ = ['limit', 'series']


def _abstract(expr: Expr, facts: Facts, x: Symbol) -> tuple[Expr, dict[Basic, Basic]]:
    """The parameters replaced by dummies carrying their assumptions."""
    refiner = _Refiner(facts)
    forward: dict[Basic, Basic] = {}
    back: dict[Basic, Basic] = {}
    for parameter in sorted((s for s in free_symbols(expr) if s != x), key=lambda s: s.name):
        if _evaluate(normalize(as_boolean_(Eq(parameter, 0))), facts) is True:
            # a parameter known to vanish is replaced by zero
            forward[parameter] = S.Zero
            continue
        dummy = refiner.dummy_for(parameter)
        if dummy is not None:
            forward[parameter] = dummy
            back[dummy] = parameter
    return as_expr(expr.xreplace(forward)), back


def _sign_dependency(message: str, back: dict[Basic, Basic]) -> Optional[Expr]:
    """The expression whose sign SymPy needs, from its error message."""
    found = re.search(r"sign of (.*)$", message)
    if not found:
        return None
    text = found.group(1).strip()
    try:
        e = parse_expr(text, local_dict={str(d): d for d in back})
    except (ValueError, TypeError, SyntaxError, NameError):
        return None
    if isinstance(e, sign):
        e = e.args[0]
    if not isinstance(e, Expr):
        return None
    return as_expr(e.xreplace(back))


def _cases(e: Expr, facts: Facts, assumptions: Assumptions) -> list[Boolean]:
    """The sign cases of ``e`` compatible with the assumptions."""
    result: list[Boolean] = []
    for case in (e > 0, Eq(e, 0), e < 0):
        case_ = normalize(as_boolean_(case))
        value = _evaluate(case_, facts)
        if value is False:
            continue
        if value is True:
            return [true]
        if satisfiable(case_, assumptions) is False:
            continue
        result.append(case_)
    return result


def as_boolean_(b: object) -> Boolean:
    from sympy_extras._typing import as_boolean
    return as_boolean(b)


def limit(expr: Union[Expr, int], x: Symbol, x0: Union[Expr, int], direction: str = '+',
          assumptions: Assumptions = None, domain: Optional[Set] = None) -> Expr:
    """The limit of ``expr`` as ``x`` tends to ``x0`` under assumptions on
    the parameters (``limit(expr, x, x0, dir)`` of SymPy otherwise).

    Examples
    ========

    >>> from sympy import exp, oo, log
    >>> from sympy.abc import a, x
    >>> from sympy_extras.assumptions import limit
    >>> limit(exp(a*x), x, oo, assumptions=a < 0)
    0
    >>> limit(exp(a*x), x, oo)
    Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))
    >>> limit(x**a, x, oo, assumptions=a > 0)
    oo
    >>> limit((1 + a/x)**x, x, oo)
    exp(a)
    >>> limit(x**a*log(x), x, 0, assumptions=a > 0)
    0
    """
    expr_ = as_expr(sympify(expr))
    x0_ = as_expr(sympify(x0))
    facts = _facts(assumptions, S.Reals if domain is None else domain, free_symbols(expr_) | {x})
    return _limit_under(expr_, x, x0_, direction, facts, assumptions, depth=0)


def _limit_under(expr: Expr, x: Symbol, x0: Expr, direction: str, facts: Facts,
                 assumptions: Assumptions, depth: int) -> Expr:
    abstracted, back = _abstract(expr, facts, x)
    try:
        value = _limit(abstracted, x, x0, direction)
    except NotImplementedError as error:
        dependency = _sign_dependency(str(error), back)
        if dependency is None or depth > 3:
            return Limit(expr, x, x0, direction)
        return _split(expr, x, x0, direction, facts, assumptions, dependency, depth)
    except (ValueError, TypeError):
        return Limit(expr, x, x0, direction)
    result = as_expr(value.xreplace(back))
    # signs SymPy could not decide, and powers of infinity with a
    # parameter in the exponent, are decided from the assumptions or
    # split into cases
    undecided = _undecided(result)
    if undecided is not None and depth <= 3:
        decided = _decide_sign(undecided, facts)
        if decided is not None:
            result = as_expr(result.xreplace({sign(undecided): decided}))
            return _finish(result, undecided, decided, expr, x, x0, direction, facts, assumptions, depth)
        return _split(expr, x, x0, direction, facts, assumptions, undecided, depth)
    if isinstance(result, Limit) and depth <= 3:
        # an unevaluated limit: try the sign cases of the parameters' atoms
        for parameter in sorted((s for s in free_symbols(expr) if s != x), key=lambda s: s.name):
            cases = _cases(parameter, facts, assumptions)
            if len(cases) > 1:
                return _split(expr, x, x0, direction, facts, assumptions, parameter, depth)
    return result


def _finish(result: Expr, undecided: Expr, decided: Expr, expr: Expr, x: Symbol, x0: Expr, direction: str,
            facts: Facts, assumptions: Assumptions, depth: int) -> Expr:
    """The result with a decided sign substituted; when the substitution
    leaves other undecided parts, the corresponding case is recomputed
    with the sign as an extra assumption."""
    if _undecided(result) is None:
        return result
    case = undecided > 0 if decided == 1 else (undecided < 0 if decided == -1 else Eq(undecided, 0))
    extra: list[Union[Boolean, bool]] = [as_boolean_(case)]
    if isinstance(assumptions, (Boolean, bool)):
        extra.append(assumptions)
    elif assumptions is not None:
        extra.extend(assumptions)
    return _limit_under(expr, x, x0, direction, Facts(extra, S.Reals, list(facts.real)), And(*extra), depth + 1)


def _undecided(result: Expr) -> Optional[Expr]:
    """An expression whose sign the result depends on: the argument of a
    ``sign`` or the exponent of a power of infinity."""
    for atom in result.atoms(sign):
        return as_expr(atom.args[0])
    from sympy.core.power import Pow
    for atom in result.atoms(Pow):
        if atom.base in (S.Infinity, S.NegativeInfinity) and free_symbols(as_expr(atom.exp)):
            return as_expr(atom.exp)
    for atom in result.atoms(Mul):
        if any(a in (S.Infinity, S.NegativeInfinity) for a in atom.args) and free_symbols(atom):
            others = [as_expr(a) for a in atom.args if a not in (S.Infinity, S.NegativeInfinity)]
            if others:
                return as_expr(Mul(*others))
    return None


def _decide_sign(e: Expr, facts: Facts) -> Optional[Expr]:
    if _evaluate(normalize(as_boolean_(e > 0)), facts) is True:
        return S.One
    if _evaluate(normalize(as_boolean_(e < 0)), facts) is True:
        return S.NegativeOne
    if _evaluate(normalize(as_boolean_(Eq(e, 0))), facts) is True:
        return S.Zero
    return None


def _split(expr: Expr, x: Symbol, x0: Expr, direction: str, facts: Facts, assumptions: Assumptions,
           dependency: Expr, depth: int) -> Expr:
    pieces: list[tuple[Expr, Boolean]] = []
    for case in _cases(dependency, facts, assumptions):
        extra: list[Union[Boolean, bool]] = [case]
        if isinstance(assumptions, (Boolean, bool)):
            extra.append(assumptions)
        elif assumptions is not None:
            extra.extend(assumptions)
        case_facts = Facts(extra, S.Reals, list(facts.real))
        if case is true:
            return _limit_under(expr, x, x0, direction, case_facts, And(*extra), depth + 1)
        value = _limit_under(expr, x, x0, direction, case_facts, And(*extra), depth + 1)
        pieces.append((value, case))
    if not pieces:
        return Limit(expr, x, x0, direction)
    if len(pieces) == 1:
        return pieces[0][0]
    return as_expr(Piecewise(*pieces))


def series(expr: Union[Expr, int], x: Symbol, x0: Union[Expr, int] = 0, n: int = 6,
           assumptions: Assumptions = None, domain: Optional[Set] = None) -> Expr:
    """The series expansion of ``expr`` about ``x0`` with the parameters
    carrying the assumptions.

    Examples
    ========

    >>> from sympy import sqrt, log
    >>> from sympy.abc import a, x
    >>> from sympy_extras.assumptions import series
    >>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a > 0)
    x/(2*a) + a + O(x**2)
    >>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0)
    -x/(2*a) - a + O(x**2)
    >>> series(log(a*x), x, 1, 2, assumptions=a > 0)
    -1 + log(a) + x + O((x - 1)**2, (x, 1))
    """
    expr_ = as_expr(sympify(expr))
    x0_ = as_expr(sympify(x0))
    facts = _facts(assumptions, S.Reals if domain is None else domain, free_symbols(expr_) | {x})
    abstracted, back = _abstract(expr_, facts, x)
    expansion = abstracted.series(x, x0_, n)
    result = as_expr(expansion.xreplace(back))
    return result
