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
``x`` and the conjugates. Also ``jacobi``, ``gegenbauer``,
``assoc_legendre`` and ``hermite_prob`` with SymPy's reductions
(``jacobi(n, a, a, x)`` through ``gegenbauer``, ``chebyshevt``,
``legendre``, ``chebyshevu``; ``jacobi(n, a, -a, x)`` through
``assoc_legendre``). Not ported: ``jacobi`` at ``x = 0`` (SymPy's answer
is a ``hyper``), the derivatives in the parameters (SymPy's are
``Sum``), the Chebyshev roots, the rewrites.

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
# chebyshevu, hermite, laguerre, assoc_laguerre, jacobi, gegenbauer, assoc_legendre,
# hermite_prob) and the recurrences of
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

__all__ = ['Legendre', 'ChebyshevT', 'ChebyshevU', 'Hermite', 'Laguerre', 'AssocLaguerre', 'Jacobi', 'Gegenbauer',
           'AssocLegendre', 'HermiteProb', 'legendre', 'chebyshevt', 'chebyshevu', 'hermite', 'laguerre', 'assoc_laguerre',
           'jacobi', 'gegenbauer', 'assoc_legendre', 'hermite_prob']

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


def _lmul(*factors: Expr) -> Expr:
    """The product evaluated left to right, as Python evaluates SymPy's
    ``a*b/c*d``."""
    result = factors[0]
    for f in factors[1:]:
        result = mul(result, f)
    return result


def _hermite_prob(n: int) -> Coefficients:
    # He_{k+1} = x*He_k - k*He_{k-1}
    return _recurrence(n, [ONE], [ZERO, ONE], lambda k, p, q: _combine((ONE, _shift(p)), (Integer(-k), q)))


def _gegenbauer(n: int, a: Expr) -> Coefficients:
    # k*C_k = 2*(k + a - 1)*x*C_{k-1} - (k + 2*a - 2)*C_{k-2}
    def step(k: int, p: Coefficients, q: Coefficients) -> Coefficients:
        j = k + 1
        return _combine((mul(Rational(2, j), add(Integer(j - 1), a)), _shift(p)),
                        (mul(Rational(-1, j), add(Integer(j - 2), mul(Integer(2), a))), q))
    return _recurrence(n, [ONE], [ZERO, expand(mul(Integer(2), a))], step)


def _binomial_polynomial(top: Expr, k: int) -> Expr:
    """``binomial(top, k)`` as an expanded polynomial in ``top``."""
    from math import factorial as integer_factorial
    product: Expr = ONE
    for i in range(k):
        product = mul(product, sub(top, Integer(i)))
    return expand(mul(Rational(1, integer_factorial(k)), product))


def _power_coefficients(shift: int, power: int) -> list[Rational]:
    """The coefficients of ``((x + shift)/2)**power``."""
    from math import comb
    return [Rational(comb(power, k) * shift**(power - k), 2**power) for k in range(power + 1)]


def _jacobi(n: int, a: Expr, b: Expr) -> Coefficients:
    """``P_n^(a, b) = sum(binomial(n + a, n - s)*binomial(n + b, s)
    *((x - 1)/2)**s*((x + 1)/2)**(n - s))``, the coefficients expanded."""
    coefficients: list[list[Expr]] = [[] for _ in range(n + 1)]
    for s in range(n + 1):
        weight = mul(_binomial_polynomial(add(Integer(n), a), n - s), _binomial_polynomial(add(Integer(n), b), s))
        minus, plus = _power_coefficients(-1, s), _power_coefficients(1, n - s)
        for i, c in enumerate(minus):
            for j, d in enumerate(plus):
                coefficients[i + j].append(mul(c, d, weight))
    return [expand(add(*terms)) for terms in coefficients]


def _derivative(coefficients: Coefficients, order: int) -> Coefficients:
    result = list(coefficients)
    for _ in range(order):
        result = [mul(Integer(k), c) for k, c in enumerate(result)][1:] or [ZERO]
    return result


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:jacobi
class Jacobi(DefinedFunction):
    """The Jacobi polynomial ``jacobi(n, a, b, x)``, unevaluated
    (``jacobi`` evaluates)."""

    function_name = 'jacobi'
    nargs = frozenset({4})
    call_nargs = frozenset({4})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.combinatorial.factorials import factorial, rf
        from .gamma_functions import gamma
        n, a, b, x = args
        if a == b:
            if a == Rational(-1, 2):
                return _lmul(div(rf(HALF, n), factorial(n)), chebyshevt(n, x))
            if a.is_zero:
                return legendre(n, x)
            if a == HALF:
                return _lmul(div(rf(Rational(3, 2), n), factorial(add(n, ONE))), chebyshevu(n, x))
            return _lmul(div(rf(add(a, ONE), n), rf(add(mul(Integer(2), a), ONE), n)), gegenbauer(n, add(a, HALF), x))
        if b == mul(MINUS_ONE, a):
            half_a = div(a, Integer(2))
            return _lmul(div(gamma(add(n, a, ONE)), gamma(add(n, ONE))), pow(add(ONE, x), half_a),
                         pow(pow(sub(ONE, x), half_a), MINUS_ONE), assoc_legendre(n, mul(MINUS_ONE, a), x))
        if not n.is_Number:
            if x.could_extract_minus_sign():
                return mul(pow(MINUS_ONE, n), jacobi(n, b, a, mul(MINUS_ONE, x)))
            if x.is_zero:
                return None  # SymPy: a hyper, not ported
            if x == ONE:
                return div(rf(add(a, ONE), n), factorial(n))
            if isinstance(x, Infinity) and n.is_positive:
                if add(a, b, mul(Integer(2), n)).is_integer:
                    raise ValueError("Error. a + b + 2*n should not be an integer.")
                return mul(rf(add(a, b, n, ONE), n), oo)
            return None
        order = _integer(n)
        if order is None:
            return None
        if order < 0:
            raise ValueError("Cannot generate Jacobi polynomial of degree %s" % (n,))
        return _as_expr(_jacobi(order, a, b), x)

    def fdiff(self, argindex: int) -> Expr:
        n, a, b, x = self.args
        if argindex == 4:
            return _lmul(HALF, add(a, b, n, ONE), jacobi(sub(n, ONE), add(a, ONE), add(b, ONE), x))
        return DefinedFunction.fdiff(self, argindex)  # SymPy: a Sum, not ported

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        n, a, b, x = self.args
        return jacobi(n, conjugate(a), conjugate(b), conjugate(x))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.jacobi(*values))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:gegenbauer
class Gegenbauer(DefinedFunction):
    """The Gegenbauer polynomial ``gegenbauer(n, a, x)``, unevaluated
    (``gegenbauer`` evaluates)."""

    function_name = 'gegenbauer'
    nargs = frozenset({3})
    call_nargs = frozenset({3})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.relational import gt
        from sympy_extras.core.logic import true
        from sympy_extras.functions.combinatorial.factorials import rf
        from sympy_extras.functions.elementary.complexes import re
        from sympy_extras.functions.elementary.trigonometric import cos, sec
        from .gamma_functions import gamma
        n, a, x = args
        if n.is_negative:
            return ZERO
        if a == HALF:
            return legendre(n, x)
        if a == ONE:
            return chebyshevu(n, x)
        if a == MINUS_ONE:
            return ZERO
        if not n.is_Number:
            two_a = mul(Integer(2), a)
            if x == MINUS_ONE:
                if gt(re(a), HALF) == true:
                    from sympy_extras.core.numbers import zoo
                    return zoo
                return div(_lmul(cos(mul(pi, add(a, n))), sec(mul(pi, a)), gamma(add(two_a, n))),
                           _lmul(gamma(two_a), gamma(add(n, ONE))))
            if x.could_extract_minus_sign():
                return mul(pow(MINUS_ONE, n), gegenbauer(n, a, mul(MINUS_ONE, x)))
            if x.is_zero:
                return div(_lmul(pow(Integer(2), n), sqrt(pi), gamma(add(a, mul(HALF, n)))),
                           _lmul(gamma(div(sub(ONE, n), Integer(2))), gamma(add(n, ONE)), gamma(a)))
            if x == ONE:
                return div(gamma(add(two_a, n)), mul(gamma(two_a), gamma(add(n, ONE))))
            if isinstance(x, Infinity) and n.is_positive:
                return mul(rf(a, n), oo)
            return None
        order = _integer(n)
        if order is None:
            return None
        return _as_expr(_gegenbauer(order, a), x)

    def fdiff(self, argindex: int) -> Expr:
        n, a, x = self.args
        if argindex == 3:
            return _lmul(Integer(2), a, gegenbauer(sub(n, ONE), add(a, ONE), x))
        return DefinedFunction.fdiff(self, argindex)  # SymPy: a Sum, not ported

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        n, a, x = self.args
        return gegenbauer(n, conjugate(a), conjugate(x))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.gegenbauer(*values))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:assoc_legendre
class AssocLegendre(DefinedFunction):
    """The associated Legendre function ``assoc_legendre(n, m, x)``,
    unevaluated (``assoc_legendre`` evaluates)."""

    function_name = 'assoc_legendre'
    nargs = frozenset({3})
    call_nargs = frozenset({3})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.combinatorial.factorials import factorial
        from .gamma_functions import gamma
        n, m, x = args
        if m.could_extract_minus_sign():
            minus_m = mul(MINUS_ONE, m)
            return _lmul(pow(MINUS_ONE, minus_m), div(factorial(add(m, n)), factorial(sub(n, m))),
                         assoc_legendre(n, minus_m, x))
        if m == ZERO:
            return legendre(n, x)
        if x == ZERO:
            return div(mul(pow(Integer(2), m), sqrt(pi)),
                       mul(gamma(div(sub(sub(ONE, m), n), Integer(2))), gamma(sub(ONE, div(sub(m, n), Integer(2))))))
        nn, mm = _integer(n), _integer(m)
        if nn is not None and mm is not None:
            if nn < 0:
                raise ValueError("assoc_legendre : 1st index must be nonnegative integer (got %r)" % (n,))
            if abs(mm) > nn:
                raise ValueError("assoc_legendre : abs('2nd index') must be <= '1st index' (got %r, %r)" % (n, m))
            # SymPy builds the function in a dummy and substitutes x
            from sympy_extras.core.symbol import Dummy
            order = abs(mm)
            dummy = Dummy('x')
            derivative = _as_expr(_derivative(_legendre(nn), order), dummy)
            built = _lmul(pow(MINUS_ONE, Integer(order)), pow(sub(ONE, pow(dummy, Integer(2))), Rational(order, 2)),
                          derivative)
            substituted = built.subs(dummy, x)
            if not isinstance(substituted, Expr):
                raise TypeError("an expression is expected")
            return substituted
        return None

    def fdiff(self, argindex: int) -> Expr:
        n, m, x = self.args
        if argindex == 3:
            return _lmul(div(ONE, sub(pow(x, Integer(2)), ONE)),
                         sub(_lmul(x, n, assoc_legendre(n, m, x)), mul(add(m, n), assoc_legendre(sub(n, ONE), m, x))))
        return DefinedFunction.fdiff(self, argindex)

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        n, m, x = self.args
        return assoc_legendre(n, conjugate(m), conjugate(x))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n, m, x = values
        return mpmath.mpc(mpmath.legenp(n, m, x))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/polynomials.py:hermite_prob
class HermiteProb(OrthogonalPolynomial):
    """The probabilists' Hermite polynomial ``hermite_prob(n, x)``,
    unevaluated (``hermite_prob`` evaluates)."""

    function_name = 'hermite_prob'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .gamma_functions import gamma
        n, x = args
        order = _integer(n)
        if order is None:
            if n.is_Number:
                return None
            reflected = _minus_x(hermite_prob, n, x)
            if reflected is not None:
                return reflected
            if x.is_zero:
                return div(sqrt(pi), gamma(div(sub(ONE, n), Integer(2))))
            if isinstance(x, Infinity):
                return oo
            return None
        if order < 0:
            return None  # SymPy builds a ValueError without raising it: the node stays
        return _as_expr(_hermite_prob(order), x)

    def fdiff(self, argindex: int) -> Expr:
        n, x = self.args
        if argindex == 2:
            return mul(n, hermite_prob(sub(n, ONE), x))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n, x = values
        return mpmath.mpc(2**(-n/2)*mpmath.hermite(n, x/mpmath.sqrt(2)))


def jacobi(n: ExprLike, a: ExprLike, b: ExprLike, x: ExprLike) -> Expr:
    """``jacobi(n, a, b, x)``, evaluated."""
    return Jacobi.apply(to_expr(n), to_expr(a), to_expr(b), to_expr(x))


def gegenbauer(n: ExprLike, a: ExprLike, x: ExprLike) -> Expr:
    """``gegenbauer(n, a, x)``, evaluated."""
    return Gegenbauer.apply(to_expr(n), to_expr(a), to_expr(x))


def assoc_legendre(n: ExprLike, m: ExprLike, x: ExprLike) -> Expr:
    """``assoc_legendre(n, m, x)``, evaluated."""
    return AssocLegendre.apply(to_expr(n), to_expr(m), to_expr(x))


def hermite_prob(n: ExprLike, x: ExprLike) -> Expr:
    """``hermite_prob(n, x)``, evaluated."""
    return HermiteProb.apply(to_expr(n), to_expr(x))
