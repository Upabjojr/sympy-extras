"""``rewrite``: the nodes of an expression rewritten in terms of a
function class.

Ported from SymPy's ``Basic.rewrite`` and ``Basic._rewrite``: bottom up,
each node (of the given pattern classes, all when none) is replaced by its
rule for the target, the node rebuilt with its rewritten arguments
otherwise. The rules ported are a typed table
(:mod:`sympy_extras.functions.elementary._rewrites`); a node without one
whose SymPy class has a rule for the target is rewritten by SymPy, that
node only.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/basic.py (Basic.rewrite, Basic._rewrite) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the rules are a table keyed by (class, target).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from .basic import Basic
from .expr import Expr

__all__ = ['rewrite', 'RewriteHints']


class RewriteHints:
    """The hints of a rewrite (SymPy's keyword arguments): ``piecewise``
    lets the rules of the rising and falling factorials return a
    ``Piecewise``."""

    __slots__ = ('piecewise',)

    def __init__(self, piecewise: bool = True) -> None:
        self.piecewise = piecewise


Rule = Callable[[Expr, tuple[Expr, ...], RewriteHints], Optional[Expr]]

_RULES: dict[tuple[type, type], Rule] = {}


def _rules() -> dict[tuple[type, type], Rule]:
    if not _RULES:
        from sympy_extras.functions.combinatorial._rewrites import rules as combinatorial_rules
        from sympy_extras.functions.elementary._rewrites import rules
        _RULES.update(rules())
        _RULES.update(combinatorial_rules())
    return _RULES


def _rule_for(cls: type, target: type) -> Optional[Rule]:
    table = _rules()
    for k in cls.__mro__:
        rule = table.get((k, target))
        if rule is not None:
            return rule
    return None


def _without_rules(e: Basic) -> bool:
    """The core nodes SymPy has no rewrite rules for."""
    from .add import Add
    from .mul import Mul
    return not e.args or isinstance(e, (Add, Mul))


# PORTED FROM SYMPY: sympy/core/basic.py:Basic.rewrite
def rewrite(e: Basic, patterns: tuple[type, ...], target: type, hints: Optional[RewriteHints] = None) -> Basic:
    """``e`` with its nodes (instances of ``patterns``, all when empty)
    rewritten in terms of ``target``."""
    if patterns:
        patterns = tuple(p for p in patterns if e.has(p))
        if not patterns:
            return e
    return _rewrite(e, patterns, target, hints if hints is not None else RewriteHints())


# PORTED FROM SYMPY: sympy/core/basic.py:Basic._rewrite
def _rewrite(e: Basic, patterns: tuple[type, ...], target: type, hints: RewriteHints) -> Basic:
    from .interop import rewrite_node_through_sympy
    args = tuple(_rewrite(a, patterns, target, hints) for a in e.args)
    if not patterns or isinstance(e, patterns):
        rule = _rule_for(type(e), target)
        if rule is not None and isinstance(e, Expr):
            expr_args = tuple(a for a in args if isinstance(a, Expr))
            if len(expr_args) == len(args):
                rewritten = rule(e, expr_args, hints)
                if rewritten is not None:
                    return rewritten
        elif rule is None and not _without_rules(e):
            through = rewrite_node_through_sympy(e, args, target, hints.piecewise)
            if through is not None:
                return through
    if not args:
        return e
    return e._evaluated(args)
