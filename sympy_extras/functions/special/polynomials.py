"""Orthogonal polynomials on the typed core: ``Legendre``,
``ChebyshevT``, ``ChebyshevU``, ``Hermite``, ``Laguerre``,
``AssocLaguerre`` (unevaluated nodes) and ``legendre``, ``chebyshevt``,
``chebyshevu``, ``hermite``, ``laguerre``, ``assoc_laguerre``
(evaluating).

Ported from SymPy's ``sympy.functions.special.polynomials``: a numerical
order gives the polynomial (from the three-term recurrences, written as
SymPy's ``Poly.as_expr`` writes it: a sum of ``c_k*x**k``), negative
orders are reflected (``legendre(-n, x) = legendre(n - 1, x)``,
``chebyshevu(-n, x) = -chebyshevu(n - 2, x)``, ``laguerre(-n, x) =
exp(x)*laguerre(n - 1, -x)``); a symbolic order keeps the node but takes
out ``-x`` and evaluates at ``0``, ``1`` and ``oo``; the derivatives in
``x`` and the conjugates. Not ported: ``jacobi``, ``gegenbauer``,
``assoc_legendre``, ``hermite_prob``, the Chebyshev roots, the rewrites.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions.special.polynomials import legendre, chebyshevt, hermite, laguerre, assoc_laguerre
>>> x, n, a = Symbol('x'), Symbol('n'), Symbol('a')
>>> legendre(3, x), chebyshevt(4, x), hermite(3, x), laguerre(2, x), legendre(-3, x)
(5*x**3/2 - 3*x/2, 8*x**4 - 8*x**2 + 1, 8*x**3 - 12*x, x**2/2 - 2*x + 1, 3*x**2/2 - 1/2)
>>> legendre(n, -x), chebyshevt(n, 0), assoc_laguerre(2, a, x), legendre(n, x).diff(x)
((-1)**n*legendre(n, x), cos(pi*n/2), a**2/2 + 3*a/2 + x**2/2 + x*(-a - 2) + 1, n*(x*legendre(n, x) - legendre(n - 1, x))/(x**2 - 1))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/polynomials.py (legendre, chebyshevt,
# chebyshevu, hermite, laguerre, assoc_laguerre) and the recurrences of
# sympy/polys/orthopolys.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing,
# the polynomials from the recurrences on coefficient lists of the core;
# jacobi, gegenbauer, assoc_legendre, hermite_prob not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, expand
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import Infinity, Integer, NegativeInfinity, Rational, oo, pi
from sympy_extras.core.power import pow, sqrt

__all__ = ['Legendre', 'ChebyshevT', 'ChebyshevU', 'Hermite', 'Laguerre', 'AssocLaguerre', 'legendre', 'chebyshevt',
           'chebyshevu', 'hermite', 'laguerre', 'assoc_laguerre']

ONE = Integer(1)
ZERO = Integer(0)
MINUS_ONE = Integer(-1)
HALF = Rational(1, 2)

Coefficients = list[Expr]
Step = Callable[[int, Coefficients, Coefficients], Coefficients]


def _shift(p: Coefficients) -> Coefficients:
    """``x*p``."""
    return [ZERO] + p


def _combine(*terms: tuple[Expr, Coefficients]) -> Coefficients:
    """``sum(c*p)``, the coefficients expanded."""
    size = max(len(p) for _, p in terms)
    result: Coefficients = []
    for k in range(size):
        result.append(expand(add(*[mul(c, p[k]) for c, p in terms if k < len(p)])))
    return result


def _recurrence(n: int, first: Coefficients, second: Coefficients, step: Step) -> Coefficients:
    """``p_n`` from ``p_0``, ``p_1`` and ``p_{k+1} = step(k, p_k, p_{k-1})``."""
    if n == 0:
        return first
    previous, current = first, second
    for k in range(1, n):
        previous, current = current, step(k, current, previous)
    return current


def _as_expr(coefficients: Coefficients, x: Expr) -> Expr:
    """SymPy's ``Poly.as_expr``: ``sum(c_k*x**k)``."""
    return add(*[mul(c, pow(x, Integer(k))) for k, c in enumerate(coefficients) if c != 0])


def _legendre(n: int) -> Coefficients:
    # (k + 1)*P_{k+1} = (2k + 1)*x*P_k - k*P_{k-1}
    return _recurrence(n, [ONE], [ZERO, ONE], lambda k, p, q: _combine(
        (Rational(2*k + 1, k + 1), _shift(p)), (Rational(-k, k + 1), q)))


def _chebyshev(n: int, second: Coefficients) -> Coefficients:
    # p_{k+1} = 2*x*p_k - p_{k-1}
    return _recurrence(n, [ONE], second, lambda k, p, q: _combine((Integer(2), _shift(p)), (MINUS_ONE, q)))


def _hermite(n: int) -> Coefficients:
    # H_{k+1} = 2*x*H_k - 2*k*H_{k-1}
    return _recurrence(n, [ONE], [ZERO, Integer(2)], lambda k, p, q: _combine(
        (Integer(2), _shift(p)), (Integer(-2*k), q)))


def _laguerre(n: int, alpha: Expr) -> Coefficients:
    # (k + 1)*L_{k+1} = (2k + 1 + alpha - x)*L_k - (k + alpha)*L_{k-1}
    def step(k: int, p: Coefficients, q: Coefficients) -> Coefficients:
        inverse = Rational(1, k + 1)
        return _combine((mul(inverse, add(Integer(2*k + 1), alpha)), p), (mul(MINUS_ONE, inverse), _shift(p)),
                        (mul(MINUS_ONE, inverse, add(Integer(k), alpha)), q))
    return _recurrence(n, [ONE], [expand(add(ONE, alpha)), MINUS_ONE], step)


def _integer(n: Expr) -> Optional[int]:
    return n.p if isinstance(n, Rational) and n.q == 1 else None


class OrthogonalPolynomial(DefinedFunction):
    """The common conjugate ``f(n, conjugate(x))``."""

    nargs = frozenset({2})
    call_nargs = frozenset({2})

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        n, x = self.args
        return type(self).apply(n, conjugate(x))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def _minus_x(cls_function: Callable[[Expr, Expr], Expr], n: Expr, x: Expr) -> Optional[Expr]:
    """``(-1)**n*f(n, -x)`` when ``x`` has a minus sign."""
    if x.could_extract_minus_sign():
        return mul(pow(MINUS_ONE, n), cls_function(n, mul(MINUS_ONE, x)))
    return None


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:legendre
class Legendre(OrthogonalPolynomial):
    """The Legendre polynomial ``legendre(n, x)``, unevaluated
    (``legendre`` evaluates)."""

    function_name = 'legendre'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .gamma_functions import gamma
        n, x = args
        order = _integer(n)
        if order is None:
            if n.is_Number:
                return None
            reflected = _minus_x(legendre, n, x)
            if reflected is not None:
                return reflected
            minus_n = mul(MINUS_ONE, n)
            if n.could_extract_minus_sign() and not sub(minus_n, ONE).could_extract_minus_sign():
                return legendre(sub(minus_n, ONE), x)
            if x.is_zero:
                return div(sqrt(pi), mul(gamma(sub(HALF, div(n, Integer(2)))), gamma(add(ONE, div(n, Integer(2))))))
            if x == ONE:
                return ONE
            if isinstance(x, Infinity):
                return oo
            return None
        if order < 0:
            order = -order - 1
        return _as_expr(_legendre(order), x)

    def fdiff(self, argindex: int) -> Expr:
        n, x = self.args
        if argindex == 2:
            return mul(div(n, sub(pow(x, Integer(2)), ONE)), sub(mul(x, legendre(n, x)), legendre(sub(n, ONE), x)))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.legendre(values[0], values[1]))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:chebyshevt
class ChebyshevT(OrthogonalPolynomial):
    """The Chebyshev polynomial of the first kind ``chebyshevt(n, x)``,
    unevaluated (``chebyshevt`` evaluates)."""

    function_name = 'chebyshevt'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.trigonometric import cos
        n, x = args
        order = _integer(n)
        if order is None:
            if n.is_Number:
                return None
            reflected = _minus_x(chebyshevt, n, x)
            if reflected is not None:
                return reflected
            if n.could_extract_minus_sign():
                return chebyshevt(mul(MINUS_ONE, n), x)
            if x.is_zero:
                return cos(mul(HALF, pi, n))
            if x == ONE:
                return ONE
            if isinstance(x, Infinity):
                return oo
            return None
        return _as_expr(_chebyshev(abs(order), [ZERO, ONE]), x)

    def fdiff(self, argindex: int) -> Expr:
        n, x = self.args
        if argindex == 2:
            return mul(n, chebyshevu(sub(n, ONE), x))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.chebyt(values[0], values[1]))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:chebyshevu
class ChebyshevU(OrthogonalPolynomial):
    """The Chebyshev polynomial of the second kind ``chebyshevu(n, x)``,
    unevaluated (``chebyshevu`` evaluates)."""

    function_name = 'chebyshevu'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.trigonometric import cos
        n, x = args
        order = _integer(n)
        if order is None:
            if n.is_Number:
                return None
            reflected = _minus_x(chebyshevu, n, x)
            if reflected is not None:
                return reflected
            if n.could_extract_minus_sign():
                if n == MINUS_ONE:
                    return ZERO
                shifted = sub(mul(MINUS_ONE, n), Integer(2))
                if not shifted.could_extract_minus_sign():
                    return mul(MINUS_ONE, chebyshevu(shifted, x))
            if x.is_zero:
                return cos(mul(HALF, pi, n))
            if x == ONE:
                return add(ONE, n)
            if isinstance(x, Infinity):
                return oo
            return None
        if order < 0:
            if order == -1:
                return ZERO
            return mul(MINUS_ONE, _as_expr(_chebyshev(-order - 2, [ZERO, Integer(2)]), x))
        return _as_expr(_chebyshev(order, [ZERO, Integer(2)]), x)

    def fdiff(self, argindex: int) -> Expr:
        n, x = self.args
        if argindex == 2:
            return div(sub(mul(add(n, ONE), chebyshevt(add(n, ONE), x)), mul(x, chebyshevu(n, x))),
                       sub(pow(x, Integer(2)), ONE))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.chebyu(values[0], values[1]))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:hermite
class Hermite(OrthogonalPolynomial):
    """The (physicists') Hermite polynomial ``hermite(n, x)``, unevaluated
    (``hermite`` evaluates)."""

    function_name = 'hermite'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .gamma_functions import gamma
        n, x = args
        order = _integer(n)
        if order is None:
            if n.is_Number:
                return None
            reflected = _minus_x(hermite, n, x)
            if reflected is not None:
                return reflected
            if x.is_zero:
                return div(mul(pow(Integer(2), n), sqrt(pi)), gamma(div(sub(ONE, n), Integer(2))))
            if isinstance(x, Infinity):
                return oo
            return None
        if order < 0:
            raise ValueError("The index n must be nonnegative integer (got %r)" % (n,))
        return _as_expr(_hermite(order), x)

    def fdiff(self, argindex: int) -> Expr:
        n, x = self.args
        if argindex == 2:
            return mul(Integer(2), n, hermite(sub(n, ONE), x))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.hermite(values[0], values[1]))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:laguerre
class Laguerre(OrthogonalPolynomial):
    """The Laguerre polynomial ``laguerre(n, x)``, unevaluated
    (``laguerre`` evaluates)."""

    function_name = 'laguerre'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.exponential import exp
        n, x = args
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        order = _integer(n)
        if order is None:
            if n.is_Number:
                return None
            minus_n = mul(MINUS_ONE, n)
            if n.could_extract_minus_sign() and not sub(minus_n, ONE).could_extract_minus_sign():
                return mul(exp(x), laguerre(sub(minus_n, ONE), mul(MINUS_ONE, x)))
            if x.is_zero:
                return ONE
            if isinstance(x, NegativeInfinity):
                return oo
            if isinstance(x, Infinity):
                return mul(pow(MINUS_ONE, n), oo)
            return None
        if order < 0:
            return mul(exp(x), laguerre(Integer(-order - 1), mul(MINUS_ONE, x)))
        return _as_expr(_laguerre(order, ZERO), x)

    def fdiff(self, argindex: int) -> Expr:
        n, x = self.args
        if argindex == 2:
            return mul(MINUS_ONE, assoc_laguerre(sub(n, ONE), ONE, x))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.laguerre(values[0], 0, values[1]))


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:assoc_laguerre
class AssocLaguerre(DefinedFunction):
    """The generalised Laguerre polynomial ``assoc_laguerre(n, alpha, x)``,
    unevaluated (``assoc_laguerre`` evaluates)."""

    function_name = 'assoc_laguerre'
    nargs = frozenset({3})
    call_nargs = frozenset({3})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.combinatorial.factorials import binomial
        n, alpha, x = args
        if alpha.is_zero:
            return laguerre(n, x)
        order = _integer(n)
        if order is None:
            if n.is_Number:
                return None
            if x.is_zero:
                return binomial(add(n, alpha), alpha)
            if isinstance(x, Infinity) and n.is_positive:
                return mul(pow(MINUS_ONE, n), oo)
            if isinstance(x, NegativeInfinity) and n.is_positive:
                return oo
            return None
        if order < 0:
            raise ValueError("The index n must be nonnegative integer (got %r)" % (n,))
        return _as_expr(_laguerre(order, alpha), x)

    def fdiff(self, argindex: int) -> Expr:
        n, alpha, x = self.args
        if argindex == 3:
            return mul(MINUS_ONE, assoc_laguerre(sub(n, ONE), add(alpha, ONE), x))
        return DefinedFunction.fdiff(self, argindex)  # SymPy: a Sum in alpha, not ported

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        n, alpha, x = self.args
        return assoc_laguerre(n, conjugate(alpha), conjugate(x))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.laguerre(values[0], values[1], values[2]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def legendre(n: ExprLike, x: ExprLike) -> Expr:
    """``legendre(n, x)``, evaluated."""
    return Legendre.apply(to_expr(n), to_expr(x))


def chebyshevt(n: ExprLike, x: ExprLike) -> Expr:
    """``chebyshevt(n, x)``, evaluated."""
    return ChebyshevT.apply(to_expr(n), to_expr(x))


def chebyshevu(n: ExprLike, x: ExprLike) -> Expr:
    """``chebyshevu(n, x)``, evaluated."""
    return ChebyshevU.apply(to_expr(n), to_expr(x))


def hermite(n: ExprLike, x: ExprLike) -> Expr:
    """``hermite(n, x)``, evaluated."""
    return Hermite.apply(to_expr(n), to_expr(x))


def laguerre(n: ExprLike, x: ExprLike) -> Expr:
    """``laguerre(n, x)``, evaluated."""
    return Laguerre.apply(to_expr(n), to_expr(x))


def assoc_laguerre(n: ExprLike, alpha: ExprLike, x: ExprLike) -> Expr:
    """``assoc_laguerre(n, alpha, x)``, evaluated."""
    return AssocLaguerre.apply(to_expr(n), to_expr(alpha), to_expr(x))
