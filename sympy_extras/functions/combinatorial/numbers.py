"""Combinatorial numbers on the typed core: ``Catalan``, ``Fibonacci``,
``Lucas``, ``Harmonic``, ``Bernoulli`` (unevaluated) and ``catalan``,
``fibonacci``, ``lucas``, ``harmonic``, ``bernoulli`` (evaluating).

Ported from SymPy's ``sympy.functions.combinatorial.numbers.catalan``:
``catalan(n) = 4**n*gamma(n + 1/2)/(gamma(1/2)*gamma(n + 2))`` at the
nonnegative integers and the negative non-integers, ``-1/2`` at ``-1``,
``0`` at the integers below ``-1``; the derivative, facts and mpmath
values. ``fibonacci(n)`` and the Fibonacci polynomials
``fibonacci(n, x)``, ``lucas(n)``, the harmonic numbers ``harmonic(n,
m)`` (exact at the integers, through the Bernoulli polynomials for a
nonpositive integer ``m``, the limits at ``oo``), ``bernoulli(n)`` (SymPy's
convention ``B_1 = 1/2``) and the Bernoulli polynomials ``bernoulli(n,
x)``. Not ported: ``euler``, ``bell``, ``genocchi``, ``andre``,
``partition``, ``tribonacci`` and the other numbers of the module.

Examples
========

>>> from sympy_extras.core import Rational, Symbol
>>> from sympy_extras.functions.combinatorial.numbers import catalan
>>> catalan(5), catalan(0), catalan(-1), catalan(-3), catalan(Rational(-1, 2)), catalan(Rational(1, 2))
(42, 1, -1/2, 0, zoo, catalan(1/2))
>>> from sympy_extras.functions.combinatorial.numbers import fibonacci, lucas, harmonic, bernoulli
>>> x, n = Symbol('x'), Symbol('n')
>>> fibonacci(10), fibonacci(-5), fibonacci(4, x), lucas(6), harmonic(4), harmonic(3, 2), harmonic(n, -1)
(55, 5, x**3 + 2*x, 18, 25/12, 49/36, -n/2 + (n + 1)**2/2 - 1/2)
>>> bernoulli(1), bernoulli(6), bernoulli(3, x), bernoulli(n, 1)
(1/2, 1/42, x**3 - 3*x**2/2 + x/2, bernoulli(n))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py (catalan, fibonacci, lucas,
# harmonic, bernoulli) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: an unevaluating class with an evaluating function, strict
# typing; the other numbers of the module not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import pow

__all__ = ['Catalan', 'Fibonacci', 'Lucas', 'Harmonic', 'Bernoulli', 'catalan', 'fibonacci', 'lucas', 'harmonic',
           'bernoulli']

HALF = Rational(1, 2)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:catalan
class Catalan(DefinedFunction):
    """The Catalan numbers ``catalan(n)``, unevaluated (``catalan``
    evaluates). Not the constant ``Catalan`` of the core, which is
    :data:`sympy_extras.core.Catalan`."""

    function_name = 'catalan'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.special.gamma_functions import gamma
        [n] = args
        if (isinstance(n, Rational) and n.q == 1 and n.p >= 0) or (n.is_noninteger and n.is_negative):
            return div(mul(pow(Integer(4), n), gamma(add(n, HALF))), mul(gamma(HALF), gamma(add(n, Integer(2)))))
        if n.is_integer and n.is_negative:
            if add(n, Integer(1)).is_negative:
                return Integer(0)
            if add(n, Integer(1)).is_zero:
                return Rational(-1, 2)
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.elementary.exponential import log
        from sympy_extras.functions.special.gamma_functions import polygamma
        n = self.args[0]
        return mul(catalan(n), add(sub(polygamma(Integer(0), add(n, HALF)), polygamma(Integer(0), add(n, Integer(2)))),
                                   log(Integer(4))))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        try:
            return mpmath.mpc(4**n*mpmath.gamma(n + 0.5)/(mpmath.gamma(0.5)*mpmath.gamma(n + 2)))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        n = self.args[0]
        a = facts_of(n)
        facts: dict[str, bool] = {'commutative': True}
        if a.get('integer') and a.get('nonnegative'):
            facts['integer'] = True
        if a.get('nonnegative'):
            facts['positive'] = True
        if a.get('integer') and sub(n, Integer(3)).is_positive:
            facts['composite'] = True
        return facts


def catalan(n: ExprLike) -> Expr:
    """The Catalan number ``catalan(n)``, evaluated."""
    return Catalan.apply(to_expr(n))


def _integer(n: Expr) -> Optional[int]:
    return n.p if isinstance(n, Rational) and n.q == 1 else None


def _fibonacci_number(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:fibonacci
class Fibonacci(DefinedFunction):
    """The Fibonacci numbers ``fibonacci(n)`` and polynomials
    ``fibonacci(n, x)``, unevaluated (``fibonacci`` evaluates)."""

    function_name = 'fibonacci'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.numbers import Infinity, oo
        n = args[0]
        if isinstance(n, Infinity):
            return oo
        k = _integer(n)
        if k is None:
            return None
        if len(args) == 1:
            if k < 0:
                return mul(pow(Integer(-1), Integer(k + 1)), fibonacci(Integer(-k)))
            return Integer(_fibonacci_number(k))
        if k < 1:
            raise ValueError("Fibonacci polynomials are defined only for positive integer indices.")
        x = args[1]
        # F_1 = 1, F_2 = x, F_k = x*F_{k-1} + F_{k-2}, written as a polynomial in x
        previous: list[int] = [1]
        current: list[int] = [0, 1]
        if k == 1:
            current = previous
        for _ in range(k - 2):
            shifted = [0] + current
            following = [shifted[i] + (previous[i] if i < len(previous) else 0) for i in range(len(shifted))]
            previous, current = current, following
        return add(*[mul(Integer(c), pow(x, Integer(i))) for i, c in enumerate(current) if c])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if len(values) != 1:
            return None
        return mpmath.mpc(mpmath.fibonacci(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:lucas
class Lucas(DefinedFunction):
    """The Lucas numbers ``lucas(n)``, unevaluated (``lucas``
    evaluates)."""

    function_name = 'lucas'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.numbers import Infinity, oo
        [n] = args
        if isinstance(n, Infinity):
            return oo
        if _integer(n) is not None:
            return add(fibonacci(add(n, Integer(1))), fibonacci(sub(n, Integer(1))))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        return mpmath.mpc(mpmath.fibonacci(n + 1) + mpmath.fibonacci(n - 1))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic
class Harmonic(DefinedFunction):
    """The harmonic numbers ``harmonic(n)`` and ``harmonic(n, m) =
    sum(1/k**m, k = 1..n)``, unevaluated (``harmonic`` evaluates)."""

    function_name = 'harmonic'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from fractions import Fraction
        from sympy_extras.core.numbers import Infinity, nan, number_from_fraction, oo, zoo
        from sympy_extras.core.relational import gt, le
        from sympy_extras.core.logic import true
        from sympy_extras.functions.special.zeta_functions import zeta
        n = args[0]
        given: Optional[Expr] = args[1] if len(args) > 1 else None
        if given is not None and given == Integer(1):
            return harmonic(n)
        m: Expr = Integer(1) if given is None else given
        if n.is_zero:
            return Integer(0)
        if m.is_zero:
            return n
        if isinstance(n, Infinity):
            if m.is_negative:
                return nan
            if le(m, Integer(1)) == true:
                return oo
            if gt(m, Integer(1)) == true:
                return zeta(m)
            return None
        order = _integer(m)
        if order is not None and order <= 0:
            degree = Integer(1 - order)
            return div(sub(bernoulli(degree, add(n, Integer(1))), bernoulli(degree)), degree)
        k = _integer(n)
        if k is None:
            return None
        if k < 0 and (m.is_integer is False or m.is_nonpositive is False):
            return zoo if m == Integer(1) else nan
        if k >= 0:
            if order is not None:
                return number_from_fraction(sum((Fraction(1) / Fraction(j)**order for j in range(1, k + 1)),
                                                Fraction(0)))
            return add(*[pow(Integer(j), mul(Integer(-1), m)) for j in range(1, k + 1)])
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.special.zeta_functions import zeta
        n = self.args[0]
        m = self.args[1] if len(self.args) > 1 else Integer(1)
        if argindex == 1:
            return mul(m, zeta(add(m, Integer(1)), add(n, Integer(1))))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        m = values[1] if len(values) > 1 else mpmath.mpc(1)
        try:
            if m == 1:
                return mpmath.mpc(mpmath.harmonic(n))
            return mpmath.mpc(mpmath.zeta(m) - mpmath.zeta(m, n + 1))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:bernoulli
class Bernoulli(DefinedFunction):
    """The Bernoulli numbers ``bernoulli(n)`` (``B_1 = 1/2``) and
    polynomials ``bernoulli(n, x)``, unevaluated (``bernoulli``
    evaluates)."""

    function_name = 'bernoulli'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.numbers import Rational as R, nan, number_from_fraction
        from sympy_extras.functions.special.zeta_functions import bernoulli_number, bernoulli_polynomial
        n = args[0]
        x: Optional[Expr] = args[1] if len(args) > 1 else None
        if x is not None and x == Integer(1):
            return bernoulli(n)
        if n.is_zero:
            return Integer(1)
        if n.is_integer is False or n.is_nonnegative is False:
            if x is not None and isinstance(x, R) and x.q == 1 and x.p <= 0:
                return nan
            return None
        k = _integer(n)
        if x is None:
            if n == Integer(1):
                return R(1, 2)
            if n.is_odd and sub(n, Integer(1)).is_positive:
                return Integer(0)
            if k is not None:
                return number_from_fraction(bernoulli_number(k))
            return None
        if k is not None:
            return bernoulli_polynomial(k, x)
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        if mpmath.im(n) != 0 or mpmath.re(n) != int(mpmath.re(n)) or mpmath.re(n) < 0:
            return None
        k = int(mpmath.re(n))
        if len(values) == 1:
            return mpmath.mpc(mpmath.bernoulli(k) if k != 1 else 0.5)
        return mpmath.mpc(mpmath.bernpoly(k, values[1]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def fibonacci(n: ExprLike, x: Optional[ExprLike] = None) -> Expr:
    """``fibonacci(n)`` or the polynomial ``fibonacci(n, x)``, evaluated."""
    if x is None:
        return Fibonacci.apply(to_expr(n))
    return Fibonacci.apply(to_expr(n), to_expr(x))


def lucas(n: ExprLike) -> Expr:
    """``lucas(n)``, evaluated."""
    return Lucas.apply(to_expr(n))


def harmonic(n: ExprLike, m: Optional[ExprLike] = None) -> Expr:
    """``harmonic(n)`` or ``harmonic(n, m)``, evaluated."""
    if m is None:
        return Harmonic.apply(to_expr(n))
    return Harmonic.apply(to_expr(n), to_expr(m))


def bernoulli(n: ExprLike, x: Optional[ExprLike] = None) -> Expr:
    """``bernoulli(n)`` or the polynomial ``bernoulli(n, x)``, evaluated."""
    if x is None:
        return Bernoulli.apply(to_expr(n))
    return Bernoulli.apply(to_expr(n), to_expr(x))
