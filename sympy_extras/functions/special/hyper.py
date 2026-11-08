"""The generalised hypergeometric function and the Meijer G-function on
the typed core: ``Hyper``, ``MeijerG`` (the nodes) and ``hyper``,
``meijerg`` (building them as SymPy's constructors do).

Ported from SymPy's ``sympy.functions.special.hyper``. The parameters are
lists, so the nodes are expressions of their own (not
``DefinedFunction``, whose arguments are expressions) with the
parameter lists as ``Tuple`` arguments, as in SymPy. ``hyper`` cancels
the parameters common to ``ap`` and ``bq`` and orders them (SymPy's
``hyper.__new__``); ``meijerg`` orders its four lists and refuses
infinite parameters and an ``a_i - b_j`` that is a positive integer.
Both have the derivative in the argument (``Derivative`` when a
parameter depends on the variable: SymPy's derivatives in the
parameters are not available either), mpmath values, and ``hyper`` its
radius of convergence. Not ported: ``hyperexpand``, the series,
``convergence_statement`` and the polar arguments. Appell's ``appellf1``
(a function of the core, its parameters being plain arguments) is here
too.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions.special.hyper import hyper, meijerg
>>> a, b, c, z = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('z')
>>> hyper((b, a), (c,), z), hyper((a, c), (c,), z), hyper((a, b), (c,), z).diff(z)
(hyper((a, b), (c,), z), hyper((a,), (), z), a*b*hyper((a + 1, b + 1), (c + 1,), z)/c)
>>> hyper((a, b), (c,), z).radius_of_convergence, meijerg((a,), (b,), (c,), (), z)
(1, meijerg(((a,), (b,)), ((c,), ()), z))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/hyper.py (hyper, meijerg, _prep_tuple,
# TupleParametersBase._eval_derivative) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed expression nodes with the parameter lists as fields,
# strict typing; hyperexpand, the series, convergence_statement, the
# polar arguments and appellf1 not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from collections import Counter
from typing import Final, Optional, Sequence, Union

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import ComplexInfinity, Infinity, Integer, NegativeInfinity, Rational, oo
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.sorting import ClassKey, default_sort_key
from sympy_extras.core.symbol import Symbol

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sympy_extras.core.logic import Boolean

__all__ = ['Hyper', 'MeijerG', 'AppellF1', 'hyper', 'meijerg', 'appellf1']

Parameters = Union[Tuple, Sequence[ExprLike]]
ONE = Integer(1)


def _parameters(p: Parameters) -> tuple[Expr, ...]:
    items: Sequence[Union[Basic, ExprLike]] = p.args if isinstance(p, Tuple) else p
    result: list[Expr] = []
    for item in items:
        if isinstance(item, Basic) and not isinstance(item, Expr):
            raise TypeError("the parameters are expressions, not %s" % (item,))
        result.append(item if isinstance(item, Expr) else to_expr(item))
    return tuple(result)


def _ordered(p: Sequence[Expr]) -> tuple[Expr, ...]:
    return tuple(sorted(p, key=default_sort_key))


def _depends(parameters: Sequence[Expr], s: Symbol) -> bool:
    return any(s in a.free_symbols for a in parameters)


def _numeric_list(parameters: Sequence[Expr], prec: int) -> Optional[list[mpmath.mpc]]:
    from sympy_extras.core.evalf import numeric_value
    values: list[mpmath.mpc] = []
    for a in parameters:
        value = numeric_value(a, prec)
        if value is None:
            return None
        values.append(value)
    return values


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper
class Hyper(Expr):
    """The generalised hypergeometric function ``hyper(ap, bq, z)``, the
    node as given (``hyper`` cancels and orders the parameters)."""

    __slots__ = ('_ap', '_bq', '_z')
    #: SymPy's class is a Function
    is_Function = True

    def __init__(self, ap: Parameters, bq: Parameters, z: ExprLike) -> None:
        self._ap: Final[tuple[Expr, ...]] = _parameters(ap)
        self._bq: Final[tuple[Expr, ...]] = _parameters(bq)
        self._z: Final[Expr] = to_expr(z)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (Tuple(*self._ap), Tuple(*self._bq), self._z)

    @property
    def ap(self) -> tuple[Expr, ...]:
        return self._ap

    @property
    def bq(self) -> tuple[Expr, ...]:
        return self._bq

    @property
    def argument(self) -> Expr:
        return self._z

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 10000, 'AppliedSymPyFunction')

    def _compare_name(self) -> str:
        return 'hyper'

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        ap, bq, z = args
        if not (isinstance(ap, Tuple) and isinstance(bq, Tuple) and isinstance(z, Expr)):
            raise TypeError("hyper takes two parameter tuples and an expression")
        return hyper(ap, bq, z)

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'hyper(%s, %s, %s)' % (printer.doprint(Tuple(*self._ap)), printer.doprint(Tuple(*self._bq)),
                                      printer.doprint(self._z))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}

    # PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper.fdiff
    def derivative_in_argument(self) -> Expr:
        """``prod(ap)/prod(bq)*hyper(ap + 1, bq + 1, z)``."""
        factor = div(mul(*self._ap), mul(*self._bq))
        return mul(factor, hyper([add(a, ONE) for a in self._ap], [add(b, ONE) for b in self._bq], self._z))

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        from sympy_extras.core.function import Derivative, diff
        if _depends(self._ap + self._bq, s):
            return Derivative(self, s)
        return mul(self.derivative_in_argument(), diff(self._z, s))

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        from sympy_extras.core.evalf import numeric_value
        ap, bq, z = _numeric_list(self._ap, prec), _numeric_list(self._bq, prec), numeric_value(self._z, prec)
        if ap is None or bq is None or z is None:
            return None
        try:
            with mpmath.workprec(prec):
                return mpmath.mpc(mpmath.hyper(ap, bq, z))
        except (ValueError, ZeroDivisionError, mpmath.libmp.NoConvergence):
            return None

    # PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper.eta
    @property
    def eta(self) -> Expr:
        """``sum(ap) - sum(bq)``, which decides the convergence on the circle."""
        return sub(add(*self.ap), add(*self.bq))

    # PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper.convergence_statement
    @property
    def convergence_statement(self) -> Union[bool, 'Boolean']:
        """The condition under which the series converges (``False`` for a
        zero radius, ``True`` for an infinite one, as SymPy)."""
        from sympy_extras.core.logic import and_, or_
        from sympy_extras.core.relational import ne
        from sympy_extras.functions.elementary.complexes import abs_, re
        R = self.radius_of_convergence
        if R == 0:
            return False
        if R == oo:
            return True
        e = self.eta
        z = self.argument
        c1 = and_(re(e) < 0, abs_(z) <= 1)
        c2 = and_(0 <= re(e), re(e) < 1, abs_(z) <= 1, ne(z, 1))
        c3 = and_(re(e) >= 1, abs_(z) < 1)
        return or_(c1, c2, c3)

    # PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper.radius_of_convergence
    @property
    def radius_of_convergence(self) -> Expr:
        """The radius of convergence of the defining series: 0, 1 or oo
        (a nonpositive integer parameter can terminate it)."""
        def nonpositive_integer(a: Expr) -> bool:
            return isinstance(a, Rational) and a.q == 1 and a.p <= 0
        if any(a.is_integer and nonpositive_integer(a) for a in self._ap + self._bq):
            aints = [a.p for a in self._ap if isinstance(a, Rational) and nonpositive_integer(a)]
            bints = [b.p for b in self._bq if isinstance(b, Rational) and nonpositive_integer(b)]
            if len(aints) < len(bints):
                return Integer(0)
            popped = False
            for b in bints:
                cancelled = False
                while aints:
                    a = aints.pop()
                    if a >= b:
                        cancelled = True
                        break
                    popped = True
                if not cancelled:
                    return Integer(0)
            if aints or popped:
                return oo
        if len(self._ap) == len(self._bq) + 1:
            return ONE
        if len(self._ap) <= len(self._bq):
            return oo
        return Integer(0)


def hyper(ap: Parameters, bq: Parameters, z: ExprLike) -> Expr:
    """``hyper(ap, bq, z)`` as SymPy builds it: the parameters common to
    ``ap`` and ``bq`` cancelled, both lists ordered."""
    from sympy_extras.core.parameters import global_parameters
    top, bottom = _parameters(ap), _parameters(bq)
    if not global_parameters.evaluate:
        return Hyper(_ordered(top), _ordered(bottom), z)
    counted_top, counted_bottom = Counter(top), Counter(bottom)
    common = counted_top & counted_bottom
    counted_top -= common
    counted_bottom -= common
    new_top = [k for k in sorted(counted_top, key=default_sort_key) for _ in range(counted_top[k])]
    new_bottom = [k for k in sorted(counted_bottom, key=default_sort_key) for _ in range(counted_bottom[k])]
    return Hyper(new_top, new_bottom, z)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:meijerg
class MeijerG(Expr):
    """The Meijer G-function ``meijerg(an, aother, bm, bother, z)``, printed
    as SymPy's ``meijerg(((an), (aother)), ((bm), (bother)), z)``."""

    __slots__ = ('_an', '_aother', '_bm', '_bother', '_z')
    #: SymPy's class is a Function
    is_Function = True

    def __init__(self, an: Parameters, aother: Parameters, bm: Parameters, bother: Parameters, z: ExprLike) -> None:
        self._an: Final[tuple[Expr, ...]] = _parameters(an)
        self._aother: Final[tuple[Expr, ...]] = _parameters(aother)
        self._bm: Final[tuple[Expr, ...]] = _parameters(bm)
        self._bother: Final[tuple[Expr, ...]] = _parameters(bother)
        self._z: Final[Expr] = to_expr(z)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (Tuple(Tuple(*self._an), Tuple(*self._aother)), Tuple(Tuple(*self._bm), Tuple(*self._bother)), self._z)

    @property
    def an(self) -> tuple[Expr, ...]:
        return self._an

    @property
    def aother(self) -> tuple[Expr, ...]:
        return self._aother

    @property
    def bm(self) -> tuple[Expr, ...]:
        return self._bm

    @property
    def bother(self) -> tuple[Expr, ...]:
        return self._bother

    @property
    def argument(self) -> Expr:
        return self._z

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 10000, 'AppliedSymPyFunction')

    def _compare_name(self) -> str:
        return 'meijerg'

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        first, second, z = args
        if not (isinstance(first, Tuple) and isinstance(second, Tuple) and isinstance(z, Expr)):
            raise TypeError("meijerg takes two pairs of parameter tuples and an expression")
        lists = [first.args[0], first.args[1], second.args[0], second.args[1]]
        if not all(isinstance(t, Tuple) for t in lists):
            raise TypeError("meijerg takes two pairs of parameter tuples")
        an, aother, bm, bother = (_parameters(t) for t in lists if isinstance(t, Tuple))
        return meijerg(an, aother, bm, bother, z)

    def _sympystr(self, printer: StrPrinter) -> str:
        first = Tuple(Tuple(*self._an), Tuple(*self._aother))
        second = Tuple(Tuple(*self._bm), Tuple(*self._bother))
        return 'meijerg(%s, %s, %s)' % (printer.doprint(first), printer.doprint(second), printer.doprint(self._z))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}

    # PORTED FROM SYMPY: sympy/functions/special/hyper.py:meijerg.fdiff
    def derivative_in_argument(self) -> Expr:
        z = self._z
        if self._an:
            lowered = (sub(self._an[0], ONE),) + self._an[1:]
            g = meijerg(lowered, self._aother, self._bm, self._bother, z)
            return mul(div(ONE, z), add(mul(sub(self._an[0], ONE), self), g))
        if self._bm:
            raised = (add(self._bm[0], ONE),) + self._bm[1:]
            g = meijerg(self._an, self._aother, raised, self._bother, z)
            return mul(div(ONE, z), sub(mul(self._bm[0], self), g))
        return Integer(0)

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        from sympy_extras.core.function import Derivative, diff
        if _depends(self._an + self._aother + self._bm + self._bother, s):
            return Derivative(self, s)  # SymPy's derivatives in the parameters, not ported
        return mul(self.derivative_in_argument(), diff(self._z, s))

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        from sympy_extras.core.evalf import numeric_value
        lists = [_numeric_list(p, prec) for p in (self._an, self._aother, self._bm, self._bother)]
        z = numeric_value(self._z, prec)
        an, aother, bm, bother = lists
        if an is None or aother is None or bm is None or bother is None or z is None:
            return None
        try:
            with mpmath.workprec(prec):
                return mpmath.mpc(mpmath.meijerg([an, aother], [bm, bother], z))
        except (ValueError, ZeroDivisionError, mpmath.libmp.NoConvergence):
            return None


def meijerg(an: Parameters, aother: Parameters, bm: Parameters, bother: Parameters, z: ExprLike) -> Expr:
    """``meijerg(an, aother, bm, bother, z)`` as SymPy builds it: each list
    ordered; infinite parameters and an ``a_i - b_j`` (over ``an`` and
    ``bm``) that is a positive integer refused."""
    lists = [_ordered(_parameters(p)) for p in (an, aother, bm, bother)]
    if any(isinstance(a, (Infinity, NegativeInfinity, ComplexInfinity)) for p in lists for a in p):
        raise ValueError("G-function parameters must be finite")
    for a in lists[0]:
        for b in lists[2]:
            d = sub(a, b)
            if isinstance(d, Rational) and d.q == 1 and d.p > 0:
                raise ValueError("no parameter a1, ..., an may differ from any b1, ..., bm by a positive integer")
    return MeijerG(lists[0], lists[1], lists[2], lists[3], z)



# PORTED FROM SYMPY: sympy/functions/special/hyper.py:appellf1
from sympy_extras.core.function import DefinedFunction  # noqa: E402 (the classes above do not need it)


class AppellF1(DefinedFunction):
    """Appell's hypergeometric function ``appellf1(a, b1, b2, c, x, y)``,
    unevaluated (``appellf1`` evaluates): the symmetry ``(b1, x) <-> (b2,
    y)`` put in canonical order, ``1`` at ``x = y = 0``."""

    function_name = 'appellf1'
    nargs = frozenset({6})
    call_nargs = frozenset({6})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        a, b1, b2, c, x, y = args
        if default_sort_key(b1) > default_sort_key(b2):
            return appellf1(a, b2, b1, c, y, x)
        if b1 == b2 and default_sort_key(x) > default_sort_key(y):
            return appellf1(a, b1, b2, c, y, x)
        if x == 0 and y == 0:
            return ONE
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.core.function import Derivative
        a, b1, b2, c, x, y = self.args
        if argindex == 5:
            return mul(div(mul(a, b1), c), appellf1(add(a, ONE), add(b1, ONE), b2, add(c, ONE), x, y))
        if argindex == 6:
            return mul(div(mul(a, b2), c), appellf1(add(a, ONE), b1, add(b2, ONE), add(c, ONE), x, y))
        parameter = self.args[argindex - 1]
        if isinstance(parameter, Symbol):
            return Derivative(self, parameter)
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.appellf1(*values))
        except (ValueError, ZeroDivisionError, mpmath.libmp.NoConvergence):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def appellf1(a: ExprLike, b1: ExprLike, b2: ExprLike, c: ExprLike, x: ExprLike, y: ExprLike) -> Expr:
    """``appellf1(a, b1, b2, c, x, y)``, evaluated."""
    return AppellF1.apply(to_expr(a), to_expr(b1), to_expr(b2), to_expr(c), to_expr(x), to_expr(y))
