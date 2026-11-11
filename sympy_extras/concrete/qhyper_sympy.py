r"""The SymPy face of q-hypergeometric summation
(:mod:`sympy_extras.concrete.qhyper`).

:mod:`~sympy_extras.concrete.qhyper` is written on the core and takes and
returns core expressions. Code on SymPy objects uses this module instead:
:class:`QPochhammer` is a SymPy function (which evaluates, as SymPy's
functions do, and which :mod:`sympy_extras.interop` converts to and from
the node of the core), and :func:`qbinomial`, :func:`q_ratio`,
:func:`normal_in`, :func:`qgosper_term`, :func:`qgosper_sum` and
:func:`qzeilberger` take and return SymPy expressions, converted at the
boundary around the native functions.

Examples
========

>>> from sympy.abc import q, k, n, x
>>> from sympy_extras.concrete.qhyper_sympy import qgosper_sum, qzeilberger, qbinomial
>>> qgosper_sum(q**k, (k, 0, n), q)
Piecewise((n + 1, Eq(q, 1)), (q**(n + 1)/(q - 1) - 1/(q - 1), True))
>>> qzeilberger(qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k, n, k, q).coefficients
[-q**n*x - 1, 1]
"""
from __future__ import annotations

from typing import Final, Optional, Sequence, Union

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._typing_sympy import as_expr, as_symbol
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import expr_from_sympy, to_sympy

from . import qhyper as core_qhyper

__all__ = ['QPochhammer', 'qpochhammer', 'qbinomial', 'q_ratio', 'normal_in', 'qgosper_term',
           'qgosper_sum', 'qzeilberger', 'QTelescoper']


class QPochhammer(Function):
    """The q-Pochhammer symbol ``(a; q)_k = prod_{i=0}^{k-1} (1 - a q**i)``
    as a SymPy function, evaluated for an integer ``k`` and for ``a = 0``
    (:class:`sympy_extras.concrete.qhyper.QPochhammer` is the node of the
    core).

    >>> from sympy.abc import a, q, k
    >>> from sympy_extras.concrete.qhyper_sympy import QPochhammer
    >>> QPochhammer(a, q, 2)
    (1 - a)*(-a*q + 1)
    >>> QPochhammer(a, q, k), QPochhammer(0, q, k)
    (QPochhammer(a, q, k), 1)
    """

    @classmethod
    def eval(cls, a: Expr, q: Expr, k: Expr) -> Optional[Expr]:
        if isinstance(k, Integer) and int(k) >= 0:
            return as_expr(Mul(*[1 - a*q**i for i in range(int(k))]))
        if a == 0:
            return S.One
        if isinstance(k, Integer):
            # (a; q)_{-m} = 1/prod_{i=1}^{m} (1 - a q**(-i))
            return as_expr(1/Mul(*[1 - a*q**(-i) for i in range(1, -int(k) + 1)]))
        return None


def _core(e: Union[Basic, int]) -> CoreExpr:
    return expr_from_sympy(sympify(e))


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = _core(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy(e: CoreExpr) -> Expr:
    return as_expr(to_sympy(e))


def qpochhammer(a: Union[Expr, int], q: Union[Expr, int], k: Union[Expr, int]) -> Expr:
    """``(a; q)_k`` (see :class:`QPochhammer`)."""
    return as_expr(QPochhammer(a, q, k))


def qbinomial(n: Union[Expr, int], k: Union[Expr, int], q: Union[Expr, int]) -> Expr:
    """:func:`sympy_extras.concrete.qhyper.qbinomial` on SymPy expressions.

    >>> from sympy import factor
    >>> from sympy.abc import q
    >>> from sympy_extras.concrete.qhyper_sympy import qbinomial
    >>> factor(qbinomial(4, 2, q))
    (q**2 + 1)*(q**2 + q + 1)
    """
    return _sympy(core_qhyper.qbinomial(_core(n), _core(k), _core(q)))


def q_ratio(term: Union[Expr, int], k: Symbol, q: Symbol) -> Optional[tuple[Expr, Symbol]]:
    """:func:`sympy_extras.concrete.qhyper.q_ratio` on SymPy expressions."""
    found = core_qhyper.q_ratio(_core(term), _core_symbol(k), _core_symbol(q))
    if found is None:
        return None
    ratio, Q = found
    return _sympy(ratio), as_symbol(to_sympy(Q))


def normal_in(e: Union[Expr, int], k: Symbol, q: Symbol) -> Expr:
    """:func:`sympy_extras.concrete.qhyper.normal_in` on SymPy expressions."""
    return _sympy(core_qhyper.normal_in(_core(e), _core_symbol(k), _core_symbol(q)))


def qgosper_term(term: Union[Expr, int], k: Symbol, q: Symbol, special_values: bool = True) -> Optional[Expr]:
    """:func:`sympy_extras.concrete.qhyper.qgosper_term` on SymPy
    expressions.

    >>> from sympy.abc import q, k
    >>> from sympy_extras.concrete.qhyper_sympy import qgosper_term
    >>> qgosper_term(q**k, k, q)
    Piecewise((k, Eq(q, 1)), (q**k/(q - 1), True))
    """
    found = core_qhyper.qgosper_term(_core(term), _core_symbol(k), _core_symbol(q), special_values)
    return None if found is None else _sympy(found)


def qgosper_sum(term: Union[Expr, int], limits: Sequence[Union[Symbol, Expr, int]], q: Symbol,
                special_values: bool = True) -> Optional[Expr]:
    """:func:`sympy_extras.concrete.qhyper.qgosper_sum` on SymPy
    expressions."""
    if len(limits) != 3:
        raise ValueError("the limits of a sum are (k, lo, hi), not %s" % (limits,))
    found = core_qhyper.qgosper_sum(_core(term), [_core(e) for e in limits], _core_symbol(q), special_values)
    return None if found is None else _sympy(found)


class QTelescoper:
    """:class:`sympy_extras.concrete.qhyper.QTelescoper` on SymPy
    expressions."""

    def __init__(self, term: Expr, n: Symbol, k: Symbol, q: Symbol, coefficients: list[Expr], certificate: Expr) -> None:
        self.term: Final[Expr] = term
        self.n: Final[Symbol] = n
        self.k: Final[Symbol] = k
        self.q: Final[Symbol] = q
        self.coefficients: Final[list[Expr]] = coefficients
        self.certificate: Final[Expr] = certificate

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def recurrence(self, name: str = 'S') -> Expr:
        f = Function(name)
        return as_expr(Add(*[a*f(self.n + j) for j, a in enumerate(self.coefficients)]))

    def __repr__(self) -> str:
        return "QTelescoper(%s, %s)" % (self.coefficients, self.certificate)


def qzeilberger(term: Union[Expr, int], n: Symbol, k: Symbol, q: Symbol, max_order: int = 3) -> Optional[QTelescoper]:
    """:func:`sympy_extras.concrete.qhyper.qzeilberger` on SymPy
    expressions."""
    found = core_qhyper.qzeilberger(_core(term), _core_symbol(n), _core_symbol(k), _core_symbol(q), max_order)
    if found is None:
        return None
    return QTelescoper(as_expr(sympify(term)), n, k, q, [_sympy(c) for c in found.coefficients],
                       _sympy(found.certificate))
