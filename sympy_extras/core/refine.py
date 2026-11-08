"""``refine``: an expression simplified under assumptions.

The tree is refined bottom up; each node is simplified by the rule of its
class (``_eval_refine``: a power of a negated base with an integer
exponent, ``exp`` of rational multiples of ``pi*I``, a boolean decided by
``ask``) or by a handler (``Abs``, ``Pow``, ``atan2``, ``re``, ``im``,
``arg``, ``sign``), the result refined again. The assumptions are those
given and those of ``global_assumptions``, written with the ``Q`` of this
core; the questions go to :func:`~sympy_extras.core.ask.ask` [1]_.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: refine*,
       https://docs.sympy.org/latest/modules/assumptions/refine.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/assumptions/refine.py, and the _eval_refine of
# sympy/core/symbol.py, sympy/core/power.py, sympy/logic/boolalg.py,
# sympy/functions/elementary/exponential.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the handlers dispatch on the classes of the core; the
# MatrixElement handler is not ported (no matrices in the core).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from .basic import Basic
from .expr import Expr
from .logic import Boolean, BooleanLike, to_boolean

__all__ = ['refine']


def refine(expr: Basic, assumptions: BooleanLike = True) -> Basic:
    """``expr`` simplified under the assumptions (and the global ones).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Q, refine, sqrt, Eq
    >>> x = Symbol('x')
    >>> refine(sqrt(x**2), Q.positive(x)), refine(sqrt(x**2), Q.real(x)), refine(sqrt(x**2))
    (x, Abs(x), sqrt(x**2))
    >>> refine((-1)**(x + 1), Q.even(x)), refine(Eq(x, 0), Q.positive(x)), refine(x > 0, Q.positive(x))
    (-1, False, True)
    """
    return _refine(expr, to_boolean(assumptions))


# PORTED FROM SYMPY: sympy/assumptions/refine.py:refine
def _refine(expr: Basic, assumptions: Boolean) -> Basic:
    if expr.args:
        args = tuple(_refine(a, assumptions) for a in expr.args)
        expr = expr._evaluated(args)
    own = _eval_refine(expr, assumptions)
    if own is not None:
        return own
    handler = _handler(expr)
    if handler is None or not isinstance(expr, Expr):
        return expr
    new_expr = handler(expr, assumptions)
    if new_expr is None or expr == new_expr:
        return expr
    if not isinstance(new_expr, Expr):
        return new_expr
    return _refine(new_expr, assumptions)


def _ask(prop: Boolean, assumptions: BooleanLike = True) -> Optional[bool]:
    from .ask import ask
    return ask(prop, assumptions)


def _eval_refine(expr: Basic, assumptions: Boolean) -> Optional[Basic]:
    """The ``_eval_refine`` of the classes of SymPy."""
    from .ask import Q
    from .numbers import I, Integer, Rational, oo, nan, pi
    from .power import Pow
    from .symbol import Symbol
    from .logic import false, true
    from sympy_extras.functions.elementary.exponential import Exp
    if isinstance(expr, Symbol):
        # PORTED FROM SYMPY: sympy/core/symbol.py:Symbol._eval_refine
        return expr
    if isinstance(expr, Exp):
        # PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp._eval_refine
        # (SymPy asks without the assumptions: the global ones only)
        arg = expr.args[0]
        if arg.is_Mul:
            if arg == I*oo or arg == -(I*oo):
                return nan
            coeff = arg.as_coefficient(pi*I)
            if coeff is not None and coeff != Integer(0):
                if _ask(Q.integer(2*coeff)):
                    if _ask(Q.even(coeff)):
                        return Integer(1)
                    if _ask(Q.odd(coeff)):
                        return Integer(-1)
                    if _ask(Q.even(coeff + Rational(1, 2))):
                        return -I
                    if _ask(Q.odd(coeff + Rational(1, 2))):
                        return I
        return None
    if isinstance(expr, Pow):
        # PORTED FROM SYMPY: sympy/core/power.py:Pow._eval_refine
        b, e = expr.as_base_exp()
        if _ask(Q.integer(e), assumptions) and b.could_extract_minus_sign():
            if _ask(Q.even(e), assumptions):
                return Pow(-b, e).doit()
            if _ask(Q.odd(e), assumptions):
                return -(Pow(-b, e).doit())
        return None
    if isinstance(expr, Boolean) and not isinstance(expr, Expr):
        # PORTED FROM SYMPY: sympy/logic/boolalg.py:Boolean._eval_refine
        ret = _ask(expr, assumptions)
        if ret is True:
            return true
        if ret is False:
            return false
        return None
    return None


Handler = Callable[[Expr, Boolean], Optional[Basic]]


def _handler(expr: Basic) -> Optional[Handler]:
    """SymPy's ``handlers_dict`` by the class of the node."""
    from .power import Pow
    from sympy_extras.functions.elementary.complexes import Abs, Arg, Atan2, Im, Re, Sign
    if isinstance(expr, Abs):
        return refine_abs
    if isinstance(expr, Pow):
        return refine_Pow
    if isinstance(expr, Atan2):
        return refine_atan2
    if isinstance(expr, Re):
        return refine_re
    if isinstance(expr, Im):
        return refine_im
    if isinstance(expr, Arg):
        return refine_arg
    if isinstance(expr, Sign):
        return refine_sign
    return None


# PORTED FROM SYMPY: sympy/assumptions/refine.py:refine_abs
def refine_abs(expr: Expr, assumptions: Boolean) -> Optional[Basic]:
    from .ask import Q
    from .fuzzy import fuzzy_not
    from .mul import Mul, mul
    from sympy_extras.functions.elementary.complexes import Abs, abs_
    arg = expr.args[0]
    if not isinstance(arg, Expr):
        return None
    if _ask(Q.real(arg), assumptions) and fuzzy_not(_ask(Q.negative(arg), assumptions)):
        return arg
    if _ask(Q.negative(arg), assumptions):
        return -arg
    if isinstance(arg, Mul):
        r = [_refine(abs_(a), assumptions) for a in arg.args if isinstance(a, Expr)]
        non_abs: list[Expr] = []
        in_abs: list[Expr] = []
        for i in r:
            if isinstance(i, Abs):
                inner = i.args[0]
                in_abs.append(inner)
            elif isinstance(i, Expr):
                non_abs.append(i)
        return mul(*non_abs) * abs_(mul(*in_abs))
    return None


# PORTED FROM SYMPY: sympy/assumptions/refine.py:refine_Pow
def refine_Pow(expr: Expr, assumptions: Boolean) -> Optional[Basic]:
    from .add import Add, add
    from .ask import Q
    from .numbers import Integer, Rational
    from .power import Pow
    from sympy_extras.functions.elementary.complexes import Abs, abs_, sign
    if not isinstance(expr, Pow):
        return None
    base, exp = expr.base, expr.exp
    if isinstance(base, Abs):
        if _ask(Q.real(base.args[0]), assumptions) and _ask(Q.even(exp), assumptions):
            return base.args[0] ** exp
    if _ask(Q.real(base), assumptions):
        if base.is_number:
            if _ask(Q.even(exp), assumptions):
                return abs_(base) ** exp
            if _ask(Q.odd(exp), assumptions):
                return sign(base) * abs_(base) ** exp
        if isinstance(exp, Rational):
            if isinstance(base, Pow):
                return abs_(base.base) ** (base.exp * exp)
        if base == Integer(-1):
            if isinstance(exp, Add):
                old: Expr = expr
                coeff, terms_tuple = exp.as_coeff_add()
                terms = list(dict.fromkeys(terms_tuple))
                initial_number_of_terms = len(terms)
                even_terms = [t for t in terms if _ask(Q.even(t), assumptions)]
                odd_terms = [t for t in terms if t not in even_terms and _ask(Q.odd(t), assumptions)]
                terms = [t for t in terms if t not in even_terms and t not in odd_terms]
                if len(odd_terms) % 2:
                    new_coeff = (coeff + Integer(1)) % 2
                else:
                    new_coeff = coeff % 2
                current: Expr = expr
                if new_coeff != coeff or len(terms) < initial_number_of_terms:
                    current = base ** add(*(terms + [new_coeff]))
                if not isinstance(current, Pow):
                    return current
                e2 = 2 * current.exp
                if _ask(Q.even(e2), assumptions):
                    if e2.could_extract_minus_sign():
                        e2 = e2 * current.base
                if isinstance(e2, Add):
                    i, p = e2.as_two_terms()
                    if isinstance(p, Pow) and p.base == Integer(-1):
                        if _ask(Q.integer(p.exp), assumptions):
                            i = (i + 1) / 2
                            if _ask(Q.even(i), assumptions):
                                return current.base ** p.exp
                            if _ask(Q.odd(i), assumptions):
                                return current.base ** (p.exp + 1)
                            return current.base ** (p.exp + i)
                if old != current:
                    return current
    return None


# PORTED FROM SYMPY: sympy/assumptions/refine.py:refine_atan2
def refine_atan2(expr: Expr, assumptions: Boolean) -> Optional[Basic]:
    from .ask import Q
    from .numbers import Integer, nan, pi
    from sympy_extras.functions.elementary.trigonometric import atan
    y, x = expr.args
    if not (isinstance(y, Expr) and isinstance(x, Expr)):
        return None
    if _ask(Q.real(y) & Q.positive(x), assumptions):
        return atan(y / x)
    if _ask(Q.negative(y) & Q.negative(x), assumptions):
        return atan(y / x) - pi
    if _ask(Q.positive(y) & Q.negative(x), assumptions):
        return atan(y / x) + pi
    if _ask(Q.zero(y) & Q.negative(x), assumptions):
        return pi
    if _ask(Q.positive(y) & Q.zero(x), assumptions):
        return pi / 2
    if _ask(Q.negative(y) & Q.zero(x), assumptions):
        return -pi / 2
    if _ask(Q.zero(y) & Q.zero(x), assumptions):
        return nan
    return expr


# PORTED FROM SYMPY: sympy/assumptions/refine.py:refine_re
def refine_re(expr: Expr, assumptions: Boolean) -> Optional[Basic]:
    from .ask import Q
    from .numbers import Integer
    arg = expr.args[0]
    if not isinstance(arg, Expr):
        return None
    if _ask(Q.real(arg), assumptions):
        return arg
    if _ask(Q.imaginary(arg), assumptions):
        return Integer(0)
    return _refine_reim(expr, assumptions)


# PORTED FROM SYMPY: sympy/assumptions/refine.py:refine_im
def refine_im(expr: Expr, assumptions: Boolean) -> Optional[Basic]:
    from .ask import Q
    from .numbers import I, Integer
    arg = expr.args[0]
    if not isinstance(arg, Expr):
        return None
    if _ask(Q.real(arg), assumptions):
        return Integer(0)
    if _ask(Q.imaginary(arg), assumptions):
        return -I * arg
    return _refine_reim(expr, assumptions)


# PORTED FROM SYMPY: sympy/assumptions/refine.py:refine_arg
def refine_arg(expr: Expr, assumptions: Boolean) -> Optional[Basic]:
    from .ask import Q
    from .numbers import Integer, pi
    rg = expr.args[0]
    if not isinstance(rg, Expr):
        return None
    if _ask(Q.positive(rg), assumptions):
        return Integer(0)
    if _ask(Q.negative(rg), assumptions):
        return pi
    return None


# PORTED FROM SYMPY: sympy/assumptions/refine.py:_refine_reim
def _refine_reim(expr: Expr, assumptions: Boolean) -> Optional[Basic]:
    expanded = expr.expand(complex=True)
    if expanded != expr:
        refined = _refine(expanded, assumptions)
        if refined != expanded:
            return refined
    return None


# PORTED FROM SYMPY: sympy/assumptions/refine.py:refine_sign
def refine_sign(expr: Expr, assumptions: Boolean) -> Optional[Basic]:
    from .ask import Q
    from .numbers import I, Integer
    arg = expr.args[0]
    if not isinstance(arg, Expr):
        return None
    if _ask(Q.zero(arg), assumptions):
        return Integer(0)
    # (SymPy asks whether the argument is real or imaginary without the
    # assumptions: the global ones only)
    if _ask(Q.real(arg)):
        if _ask(Q.positive(arg), assumptions):
            return Integer(1)
        if _ask(Q.negative(arg), assumptions):
            return Integer(-1)
    if _ask(Q.imaginary(arg)):
        _, arg_im = arg.as_real_imag()
        if _ask(Q.positive(arg_im), assumptions):
            return I
        if _ask(Q.negative(arg_im), assumptions):
            return -I
    return expr
