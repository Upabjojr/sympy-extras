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
from .symbol import Symbol

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
    from .bridge import bridge
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
            through = bridge().rewrite_node(e, args, target, hints.piecewise)
            if through is not None:
                return through
    if not args:
        return e
    return e._evaluated(args)


# ---------------------------------------------------------------------
# the rewrites of the limit algorithm ('tractable', 'intractable')

def _tractable_hyperbolic(e: Expr, arg: Expr) -> Optional[Expr]:
    """SymPy's ``_eval_rewrite_as_tractable`` of the hyperbolic functions
    (in exponentials)."""
    from sympy_extras.functions.elementary.exponential import exp
    from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Csch, Sech, Sinh, Tanh
    pos, neg = exp(arg), exp(-arg)
    if isinstance(e, Sinh):
        return (pos - neg) / 2
    if isinstance(e, Cosh):
        return (pos + neg) / 2
    if isinstance(e, Tanh):
        return (pos - neg) / (pos + neg)
    if isinstance(e, Coth):
        return (pos + neg) / (pos - neg)
    if isinstance(e, (Sech, Csch)):
        # PORTED FROM SYMPY: ReciprocalHyperbolicFunction._rewrite_reciprocal
        reciprocal = Cosh(arg) if isinstance(e, Sech) else Sinh(arg)
        t = _tractable_hyperbolic(reciprocal, arg)
        if t is not None and t != reciprocal.doit():
            return 1 / t
        return None
    return None


def _log_tractable(e: Basic) -> bool:
    """Whether SymPy's tractable form of the node is its logarithmic form
    (the inverse trigonometric and hyperbolic functions)."""
    from sympy_extras.functions.elementary.hyperbolic import Acosh, Acoth, Acsch, Asech, Asinh, Atanh
    from sympy_extras.functions.elementary.trigonometric import Acos, Acot, Acsc, Asec, Asin, Atan
    return isinstance(e, (Asin, Acos, Atan, Acot, Asec, Acsc, Asinh, Acosh, Atanh, Acoth, Asech, Acsch))


# PORTED FROM SYMPY: sympy/core/basic.py:Basic._rewrite (with the rule names 'tractable', 'intractable')
def rewrite_tractable(e: Basic, limitvar: Optional[Symbol], intractable: bool = False) -> Basic:
    """``e.rewrite('tractable', deep=True, limitvar=...)`` of SymPy (the
    form of the functions with the exponentials and logarithms the limit
    algorithm works with), or ``'intractable'`` back. The elementary
    functions are rewritten here, the special ones by SymPy's rules."""
    from .bridge import bridge
    from sympy_extras.functions.elementary.exponential import Log
    from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Csch, Sech, Sinh, Tanh
    args = tuple(rewrite_tractable(a, limitvar, intractable) for a in e.args)
    if not intractable and isinstance(e, Expr):
        if isinstance(e, (Sinh, Cosh, Tanh, Coth, Sech, Csch)):
            arg = args[0]
            if isinstance(arg, Expr):
                rewritten = _tractable_hyperbolic(e, arg)
                if rewritten is not None:
                    return rewritten
        elif _log_tractable(e):
            return _rewrite(e._evaluated(args), (), Log, RewriteHints())
    name = 'intractable' if intractable else 'tractable'
    if e.args and (e._is_foreign_node() or type(e).__name__ in _SYMPY_NAMED_RULES[name]):
        # the classes with a rule of that name in SymPy, not ported: SymPy's
        through = bridge().rewrite_named_node(e, args, name, limitvar)
        if through is not None:
            return through
    if not args:
        return e
    return e._evaluated(args)


#: the classes whose SymPy counterpart has ``_eval_rewrite_as_tractable``
#: or ``_eval_rewrite_as_intractable`` (the others, sums, products and
#: powers among them, have no such rule: the node is rebuilt here)
_SYMPY_NAMED_RULES: dict[str, frozenset[str]] = {
    'tractable': frozenset({
        'Acos', 'Acosh', 'Acot', 'Acoth', 'Acsc', 'Acsch', 'Asec', 'Asech', 'Asin', 'Asinh', 'Atan', 'Atanh',
        'BesselI', 'BesselK', 'Binomial', 'Cosh', 'Coth', 'Csch', 'EiNode', 'Erf', 'Erfc', 'Erfi',
        'FallingFactorial', 'Fibonacci', 'Gamma', 'Harmonic', 'LiNode', 'OffsetLiNode', 'RisingFactorial',
        'Sech', 'Sinh', 'Tanh', 'UpperGamma'}),
    'intractable': frozenset({'LogGamma'}),
}
