"""``rewrite``: the nodes of an expression rewritten in terms of a
function class.

Ported from SymPy's ``Basic.rewrite`` and ``Basic._rewrite``: bottom up,
each node (of the given pattern classes, all when none) is replaced by its
rule for the target, the node rebuilt with its rewritten arguments
otherwise (SymPy's behaviour when a class has no rule). The rules ported
are typed tables keyed by the class of the node and the SymPy name of the
target (:mod:`sympy_extras.functions.elementary._rewrites` and its
siblings); the target is a class (a function class, ``Piecewise``,
``Integral``, ``Sum``, ...), a node (``GoldenRatio``) or a name
(``'sqrt'``, ``'polynomial'``), as in SymPy. Only a node wrapping a
foreign object is rewritten by the foreign system (the bridge).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/basic.py (Basic.rewrite, Basic._rewrite) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the rules are tables keyed by (class, target name).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional, Union

from .basic import Basic
from .expr import Expr
from .function import DefinedFunction, FunctionClass
from .symbol import Symbol

__all__ = ['rewrite', 'RewriteHints', 'RewriteTarget', 'target_name']


class RewriteHints:
    """The hints of a rewrite (SymPy's keyword arguments): ``piecewise``
    lets the rules of the rising and falling factorials return a
    ``Piecewise``."""

    __slots__ = ('piecewise',)

    def __init__(self, piecewise: bool = True) -> None:
        self.piecewise = piecewise


#: a rule of an expression: the node and its (rewritten) arguments to the
#: rewritten node, ``None`` when the rule does not apply
Rule = Callable[[Expr, tuple[Expr, ...], RewriteHints], Optional[Expr]]
#: a rule of a node which is not an expression (a boolean, a set), or
#: whose result is not one
NodeRule = Callable[[Basic, tuple[Basic, ...], RewriteHints], Optional[Basic]]
#: the target of a rewrite: a class (native or foreign), a function
#: object, a node (its class) or a SymPy name
RewriteTarget = Union[Basic, FunctionClass, type, str]
#: the key of a rule table: the class of the node and the target, a class
#: or the SymPy name of a target without a native class
RuleKey = tuple[type, Union[type, str]]

_RULES: dict[tuple[type, str], Rule] = {}
_NODE_RULES: dict[tuple[type, str], NodeRule] = {}

#: the SymPy names of the native classes whose name differs from SymPy's
_SYMPY_NAMES: dict[str, str] = {
    'Hyper': 'hyper', 'MeijerG': 'meijerg', 'AppellF1': 'appellf1', 'GoldenRatioType': 'GoldenRatio',
    'TribonacciConstantType': 'TribonacciConstant', 'PointSet': 'FiniteSet'}


def target_name(target: RewriteTarget) -> str:
    """The SymPy name of a rewrite target (what follows
    ``_eval_rewrite_as_`` in the name of SymPy's rule): the
    ``function_name`` of a function class, the SymPy name of another
    class, the class name of a node, the name itself.

    Examples
    ========

    >>> from sympy_extras.core.rewriting import target_name
    >>> from sympy_extras.functions.elementary.piecewise import Piecewise
    >>> from sympy_extras.functions.elementary.trigonometric import Cos
    >>> from sympy_extras.functions.special.hyper import Hyper
    >>> from sympy_extras.core.numbers import GoldenRatio
    >>> target_name(Cos), target_name(Piecewise), target_name(Hyper), target_name(GoldenRatio), target_name('sqrt')
    ('cos', 'Piecewise', 'hyper', 'GoldenRatio', 'sqrt')
    """
    if isinstance(target, str):
        return target
    if isinstance(target, FunctionClass):
        return target.name
    if isinstance(target, Basic):
        return _class_name(type(target))
    return _class_name(target)


def _class_name(cls: type) -> str:
    if issubclass(cls, DefinedFunction):
        return cls.function_name
    if issubclass(cls, Basic):
        return _SYMPY_NAMES.get(cls.__name__, cls.__name__)
    return cls.__name__  # a foreign class: SymPy's name


def _name_key(t: Union[type, str]) -> str:
    return t if isinstance(t, str) else _class_name(t)


def _rules() -> dict[tuple[type, str], Rule]:
    if not _RULES:
        from sympy_extras.functions.combinatorial._rewrites import rules as combinatorial_rules
        from sympy_extras.functions.elementary._rewrites import rules
        from sympy_extras.functions.special._rewrites import rules as special_rules
        from sympy_extras.functions._rewrites_misc import rules as misc_rules
        from sympy_extras.functions.special._rewrites_bessel import rules as bessel_rules
        from sympy_extras.functions.special._rewrites_polynomials import rules as polynomial_rules
        for table in (rules(), combinatorial_rules(), special_rules(), bessel_rules(), polynomial_rules(),
                      misc_rules()):
            for (cls, t), rule in table.items():
                _RULES[(cls, _name_key(t))] = rule
    return _RULES


def _node_rules() -> dict[tuple[type, str], NodeRule]:
    if not _NODE_RULES:
        from sympy_extras.functions._rewrites_misc import node_rules
        for (cls, t), rule in node_rules().items():
            _NODE_RULES[(cls, _name_key(t))] = rule
    return _NODE_RULES


def _rule_for(cls: type, target: RewriteTarget) -> Optional[Rule]:
    """The rule of an expression of class ``cls`` for the target (that of
    the first class of its MRO with one), ``None`` without one."""
    table = _rules()
    name = target_name(target)
    for k in cls.__mro__:
        rule = table.get((k, name))
        if rule is not None:
            return rule
    return None


def _node_rule_for(cls: type, target: RewriteTarget) -> Optional[NodeRule]:
    """The rule of a node which is not an expression, or into a node which
    is not one, ``None`` without one."""
    table = _node_rules()
    name = target_name(target)
    for k in cls.__mro__:
        rule = table.get((k, name))
        if rule is not None:
            return rule
    return None


def has_rule(cls: type, target: RewriteTarget) -> bool:
    """Whether the native class has a rule for the target."""
    return _rule_for(cls, target) is not None or _node_rule_for(cls, target) is not None


#: a pattern of a rewrite: a class (native: by ``isinstance``; foreign:
#: by its name), or a function object
Pattern = Union[type, FunctionClass]


def _matches(e: Basic, p: Pattern) -> bool:
    if isinstance(p, FunctionClass):
        return e.func == p
    if issubclass(p, Basic):
        return isinstance(e, p)
    # a foreign class: the nodes of that SymPy name
    if isinstance(e, DefinedFunction):
        return e.function_name == p.__name__
    f = e.func
    if isinstance(f, FunctionClass):
        return f.name == p.__name__
    return False


def _has_pattern(e: Basic, p: Pattern) -> bool:
    return _matches(e, p) or any(_has_pattern(a, p) for a in e.args)


# PORTED FROM SYMPY: sympy/core/basic.py:Basic.rewrite
def rewrite(e: Basic, patterns: tuple[Pattern, ...], target: RewriteTarget, hints: Optional[RewriteHints] = None
            ) -> Basic:
    """``e`` with its nodes (those matching ``patterns``, all when empty)
    rewritten in terms of ``target``."""
    if isinstance(target, str) and target in ('tractable', 'intractable'):
        return rewrite_tractable(e, None, target == 'intractable')
    if patterns:
        patterns = tuple(p for p in patterns if _has_pattern(e, p))
        if not patterns:
            return e
    return _rewrite(e, patterns, target, hints if hints is not None else RewriteHints())


# PORTED FROM SYMPY: sympy/core/basic.py:Basic._rewrite
def _rewrite(e: Basic, patterns: tuple[Pattern, ...], target: RewriteTarget, hints: RewriteHints) -> Basic:
    from .bridge import bridge
    args = tuple(_rewrite(a, patterns, target, hints) for a in e.args)
    if not patterns or any(_matches(e, p) for p in patterns):
        rule = _rule_for(type(e), target)
        if rule is not None:
            if isinstance(e, Expr):
                expr_args = tuple(a for a in args if isinstance(a, Expr))
                if len(expr_args) == len(args):
                    rewritten = rule(e, expr_args, hints)
                    if rewritten is not None:
                        return rewritten
        else:
            node_rule = _node_rule_for(type(e), target)
            if node_rule is not None:
                rewritten_node = node_rule(e, args, hints)
                if rewritten_node is not None:
                    return rewritten_node
            elif e.args and e._has_foreign_class():
                # a wrapped foreign node: the foreign system's rule
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
        special = _tractable_error_and_bessel(e, args, limitvar)
        if special is not None:
            return special
    if intractable and isinstance(e, Expr):
        from sympy_extras.functions.special.bessel import BesselIs, BesselKs
        from sympy_extras.functions.special.error_functions import Eis, Erfs
        from sympy_extras.functions.special.gamma_functions import LogGamma, gamma
        if isinstance(e, LogGamma) and isinstance(args[0], Expr):
            # PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_rewrite_as_intractable
            from sympy_extras.functions.elementary.exponential import log
            return log(gamma(args[0]))
        if isinstance(e, (Erfs, Eis, BesselIs, BesselKs)):
            expr_args = tuple(a for a in args if isinstance(a, Expr))
            if len(expr_args) == len(args):
                return e.rewrite_intractable(expr_args)
    name = 'intractable' if intractable else 'tractable'
    if e.args and e._is_foreign_node():
        # a wrapped foreign node: the foreign system's rule of that name
        through = bridge().rewrite_named_node(e, args, name, limitvar)
        if through is not None:
            return through
    if not args:
        return e
    return e._evaluated(args)


def _tractable_error_and_bessel(e: Expr, args: tuple[Basic, ...], limitvar: Optional[Symbol]) -> Optional[Basic]:
    """SymPy's ``_eval_rewrite_as_tractable`` of the error functions, the
    exponential and logarithmic integrals, the upper incomplete gamma
    function, ``besseli``, ``besselk`` and ``fibonacci`` (``args`` already
    rewritten, ``e`` the node before, as SymPy's ``self``); ``None`` for
    the other nodes and where SymPy's rule gives ``None``."""
    from sympy_extras.functions.special.bessel import BesselI, BesselK, besselis, besselks
    from sympy_extras.functions.special.error_functions import (EiNode, Erf, Erfc, Erfi, LiNode, OffsetLiNode, eis,
                                                                 erfs)
    from sympy_extras.functions.special.gamma_functions import UpperGamma, loggamma, lowergamma
    from sympy_extras.functions.combinatorial.numbers import Fibonacci
    from sympy_extras.functions.elementary.exponential import exp, log
    from .numbers import Integer, NegativeInfinity, oo
    expr_args = tuple(a for a in args if isinstance(a, Expr))
    if len(expr_args) != len(args):
        return None
    if isinstance(e, Erf):
        # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf._eval_rewrite_as_tractable
        from .limits import limit
        z = expr_args[0]
        if limitvar is not None and isinstance(limit(z, limitvar, oo), NegativeInfinity):
            return -1 + erfs(-z) * exp(-z**2)
        return 1 - erfs(z) * exp(-z**2)
    if isinstance(e, (Erfc, Erfi)):
        # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfc._eval_rewrite_as_tractable,
        # erfi._eval_rewrite_as_tractable (self.rewrite(erf).rewrite("tractable", deep=True))
        return rewrite_tractable(_rewrite(e, (), Erf, RewriteHints()), limitvar)
    if isinstance(e, EiNode):
        # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ei._eval_rewrite_as_tractable
        z = expr_args[0]
        return exp(z) * eis(z)
    if isinstance(e, LiNode):
        # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:li._eval_rewrite_as_tractable
        z = expr_args[0]
        return z * eis(log(z))
    if isinstance(e, OffsetLiNode):
        # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Li._eval_rewrite_as_tractable
        # (self.rewrite(li).rewrite("tractable", deep=True), without the limit variable)
        return rewrite_tractable(_rewrite(e, (), LiNode, RewriteHints()), None)
    if isinstance(e, UpperGamma):
        # PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:uppergamma._eval_rewrite_as_tractable
        s, x = expr_args
        return exp(loggamma(s)) - lowergamma(s, x)
    if isinstance(e, BesselI):
        # PORTED FROM SYMPY: sympy/functions/special/bessel.py:besseli._eval_rewrite_as_tractable
        nu, z = expr_args
        return exp(z) * besselis(nu, z) if z.is_extended_real is True else None
    if isinstance(e, BesselK):
        # PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselk._eval_rewrite_as_tractable
        nu, z = expr_args
        return exp(-z) * besselks(nu, z) if z.is_extended_real is True else None
    if isinstance(e, Fibonacci) and len(expr_args) == 1:
        # PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:fibonacci._eval_rewrite_as_tractable
        from sympy_extras.functions.elementary.trigonometric import cos
        from .numbers import GoldenRatio, pi
        from .power import sqrt
        n = expr_args[0]
        return (GoldenRatio**n - cos(pi * n) / GoldenRatio**n) / sqrt(Integer(5))
    return None


#: the classes whose SymPy counterpart has ``_eval_rewrite_as_tractable``
#: or ``_eval_rewrite_as_intractable`` (the others, sums, products and
#: powers among them, have no such rule: the node is rebuilt here; all
#: are ported, checked against SymPy by a test)
_SYMPY_NAMED_RULES: dict[str, frozenset[str]] = {
    'tractable': frozenset({
        'Acos', 'Acosh', 'Acot', 'Acoth', 'Acsc', 'Acsch', 'Asec', 'Asech', 'Asin', 'Asinh', 'Atan', 'Atanh',
        'BesselI', 'BesselK', 'Binomial', 'Cosh', 'Coth', 'Csch', 'EiNode', 'Erf', 'Erfc', 'Erfi',
        'FallingFactorial', 'Fibonacci', 'Gamma', 'Harmonic', 'LiNode', 'OffsetLiNode', 'RisingFactorial',
        'Sech', 'Sinh', 'Tanh', 'UpperGamma'}),
    'intractable': frozenset({'BesselIs', 'BesselKs', 'Eis', 'Erfs', 'LogGamma'}),
}
