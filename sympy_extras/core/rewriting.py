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
        from sympy_extras.functions.special._rewrites import rules as special_rules
        _RULES.update(rules())
        _RULES.update(combinatorial_rules())
        _RULES.update(special_rules())
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
        elif rule is None and not _without_rules(e) and _sympy_may_rewrite(e, target):
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


def _tractable_special(e: Expr, args: tuple[Basic, ...], limitvar: Optional[Symbol]) -> Optional[Basic]:
    """SymPy's ``_eval_rewrite_as_tractable`` of the gamma function and
    of the factorials written with it (``args`` already rewritten),
    ``None`` for the other nodes."""
    from sympy_extras.functions.combinatorial.factorials import Binomial, FallingFactorial, RisingFactorial
    from sympy_extras.functions.elementary.exponential import exp
    from sympy_extras.functions.special.gamma_functions import Gamma, gamma, loggamma
    from .expr import as_expr
    from .numbers import Infinity, Integer, NegativeInfinity
    from .power import pow

    def tractable(b: Basic, var: Optional[Symbol]) -> Expr:
        return as_expr(rewrite_tractable(b, var))

    if isinstance(e, Gamma) and isinstance(args[0], Expr):
        # PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:gamma._eval_rewrite_as_tractable
        return exp(loggamma(args[0]))
    if isinstance(e, (RisingFactorial, FallingFactorial)) and isinstance(args[0], Expr) \
            and isinstance(args[1], Expr):
        # PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:RisingFactorial._eval_rewrite_as_tractable,
        # FallingFactorial._eval_rewrite_as_tractable
        x, k = args[0], args[1]
        rising = isinstance(e, RisingFactorial)
        if limitvar is not None:
            k_lim = k.subs(limitvar, Infinity())
            if isinstance(k_lim, Infinity):
                if rising:
                    return tractable(gamma(x + k), limitvar) / gamma(x)
                return pow(Integer(-1), k) * tractable(gamma(k - x), limitvar) / gamma(-x)
            if isinstance(k_lim, NegativeInfinity):
                if rising:
                    return pow(Integer(-1), k) * gamma(1 - x) / tractable(gamma(-k - x + 1), limitvar)
                return gamma(x + 1) / tractable(gamma(x - k + 1), limitvar)
        return rewrite_tractable(_rewrite(e._evaluated(args), (), Gamma, RewriteHints()), limitvar)
    if isinstance(e, Binomial):
        # PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:binomial._eval_rewrite_as_tractable
        return rewrite_tractable(_rewrite(e._evaluated(args), (), Gamma, RewriteHints()), None)
    from sympy_extras.functions.combinatorial.numbers import Harmonic
    if isinstance(e, Harmonic) and all(isinstance(a, Expr) for a in args):
        # PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic._eval_rewrite_as_tractable
        # (through harmonic._eval_rewrite_as_polygamma, for a positive integer order)
        from sympy_extras.functions.special.gamma_functions import polygamma
        from .numbers import EulerGamma
        n = as_expr(args[0])
        m = as_expr(args[1]) if len(args) > 1 else Integer(1)
        if m == 1:
            return rewrite_tractable(polygamma(Integer(0), n + 1) + EulerGamma, limitvar)
        if m.is_integer is True and m.is_positive is True:
            return rewrite_tractable(pow(Integer(-1), m)*(polygamma(m - 1, Integer(1)) - polygamma(m - 1, n + 1))
                                     / gamma(m), limitvar)
    return None


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
        gamma_form = _tractable_special(e, args, limitvar)
        if gamma_form is not None:
            return gamma_form
    if intractable and isinstance(e, Expr):
        from sympy_extras.functions.special.gamma_functions import LogGamma, gamma
        if isinstance(e, LogGamma) and isinstance(args[0], Expr):
            # PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_rewrite_as_intractable
            from sympy_extras.functions.elementary.exponential import log
            return log(gamma(args[0]))
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


def _sympy_may_rewrite(e: Basic, target: type) -> bool:
    """Whether SymPy has a rule rewriting the node into ``target`` (its
    ``_eval_rewrite_as_<target>``): only then is the bridge asked. Wrapped
    foreign nodes and the classes not in the table are asked always."""
    from ._rewrite_rules import OWN_REWRITE, REWRITE_TARGETS
    from .function import DefinedFunction
    if e._is_foreign_node():
        return True
    name = type(e).__name__
    if name in OWN_REWRITE or name not in REWRITE_TARGETS:
        return True
    if not issubclass(target, DefinedFunction):
        return False  # the bridge rewrites only into functions with a SymPy class
    return target.function_name in REWRITE_TARGETS[name]
