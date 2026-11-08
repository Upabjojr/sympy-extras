"""Combinatorial numbers on the typed core: ``Catalan``, ``Fibonacci``,
``Lucas``, ``Harmonic``, ``Bernoulli``, ``Euler``, ``Bell``, ``Genocchi``,
``Partition`` (unevaluated) and ``catalan``, ``fibonacci``, ``lucas``,
``harmonic``, ``bernoulli``, ``euler``, ``bell``, ``genocchi``,
``partition`` (evaluating).

Ported from SymPy's ``sympy.functions.combinatorial.numbers.catalan``:
``catalan(n) = 4**n*gamma(n + 1/2)/(gamma(1/2)*gamma(n + 2))`` at the
nonnegative integers and the negative non-integers, ``-1/2`` at ``-1``,
``0`` at the integers below ``-1``; the derivative, facts and mpmath
values. ``fibonacci(n)`` and the Fibonacci polynomials
``fibonacci(n, x)``, ``lucas(n)``, the harmonic numbers ``harmonic(n,
m)`` (exact at the integers, through the Bernoulli polynomials for a
nonpositive integer ``m``, the limits at ``oo``), ``bernoulli(n)`` (SymPy's
convention ``B_1 = 1/2``) and the Bernoulli polynomials ``bernoulli(n,
x)``, the Euler numbers and polynomials (``euler(-1) = pi/2``), the Bell
numbers and polynomials, the Genocchi numbers and polynomials with their
facts, the partition numbers. Not ported: the incomplete Bell
polynomials ``bell(n, k, symbols)`` (a tuple argument, which the core's
functions, taking expressions, cannot hold) and ``carmichael`` (a class
of static tests in SymPy). ``tribonacci`` (numbers and polynomials),
``motzkin`` (with SymPy's shift: ``motzkin(n)`` is the Motzkin number of
``n - 1``, ``motzkin(0) = 1``) and ``andre`` are ported. The number-theoretic functions ``totient``,
``reduced_totient``, ``divisor_sigma``, ``udivisor_sigma``, ``primenu``,
``primeomega``, ``mobius``, ``primepi``, ``legendre_symbol``,
``jacobi_symbol``, ``kronecker_symbol`` are here too, on the typed
:mod:`sympy_extras.ntheory` (SymPy's ``Dict`` argument of ``totient`` is
not ported).

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
>>> from sympy_extras.functions.combinatorial.numbers import euler, bell, genocchi, partition
>>> euler(4), euler(3, x), bell(5), bell(3, x), genocchi(6), genocchi(3, x), partition(10)
(5, x**3 - 3*x**2/2 + 1/4, 52, x**3 + 3*x**2 + x, -3, -3*x**2 + 3*x, 42)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py (catalan, fibonacci, lucas,
# harmonic, bernoulli, euler, bell, genocchi, partition and the number-theoretic
# functions totient ... kronecker_symbol) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: an unevaluating class with an evaluating function, strict
# typing; the other numbers of the module not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import pow

__all__ = ['Catalan', 'Fibonacci', 'Lucas', 'Harmonic', 'Bernoulli', 'Euler', 'Bell', 'Genocchi', 'Partition', 'catalan',
           'fibonacci', 'lucas', 'harmonic', 'bernoulli', 'euler', 'bell', 'genocchi', 'partition', 'Totient',
           'ReducedTotient', 'DivisorSigma', 'UDivisorSigma', 'PrimeNu', 'PrimeOmega', 'Mobius', 'PrimePi',
           'LegendreSymbol', 'JacobiSymbol', 'KroneckerSymbol', 'totient', 'reduced_totient', 'divisor_sigma',
           'udivisor_sigma', 'primenu', 'primeomega', 'mobius', 'primepi', 'legendre_symbol', 'jacobi_symbol',
           'kronecker_symbol', 'Tribonacci', 'Motzkin', 'Andre', 'tribonacci', 'motzkin', 'andre']

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
    generic_nseries = True
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


def _bernoulli_coefficients(n: int) -> list[Fraction]:
    """The coefficients (by power) of the Bernoulli polynomial ``B_n(x)``
    with ``b_1 = -1/2``."""
    from math import comb
    from sympy_extras.functions.special.zeta_functions import bernoulli_number
    coefficients = [Fraction(0)] * (n + 1)
    for k in range(n + 1):
        b = -Fraction(1, 2) if k == 1 else bernoulli_number(k)
        coefficients[n - k] = comb(n, k) * b
    return coefficients


def _polynomial(coefficients: Sequence[Fraction], x: Expr) -> Expr:
    from sympy_extras.core.numbers import number_from_fraction
    return add(*[mul(number_from_fraction(c), pow(x, Integer(j))) for j, c in enumerate(coefficients) if c])


def _euler_coefficients(n: int) -> list[Fraction]:
    """``E_n(x) = 2/(n + 1)*(B_{n+1}(x) - 2**(n + 1)*B_{n+1}(x/2))``."""
    b = _bernoulli_coefficients(n + 1)
    return [Fraction(2, n + 1) * c * (1 - Fraction(2**(n + 1), 2**j)) for j, c in enumerate(b)][:n + 1]


def _genocchi_coefficients(n: int) -> list[Fraction]:
    """``G_n(x) = 2*(2**n*B_n(x/2) - B_n(x))``."""
    b = _bernoulli_coefficients(n)
    return [2 * c * (Fraction(2**n, 2**j) - 1) for j, c in enumerate(b)]


def _stirling2(n: int, k: int) -> int:
    from math import comb, factorial as integer_factorial
    return sum((-1)**i * comb(k, i) * (k - i)**n for i in range(k + 1)) // integer_factorial(k)


def _partitions(n: int) -> int:
    """The number of partitions of ``n`` by Euler's pentagonal recurrence."""
    p = [1] + [0] * n
    for m in range(1, n + 1):
        total, k = 0, 1
        while True:
            g1, g2 = k * (3 * k - 1) // 2, k * (3 * k + 1) // 2
            if g1 > m:
                break
            sign = 1 if k % 2 else -1
            total += sign * p[m - g1]
            if g2 <= m:
                total += sign * p[m - g2]
            k += 1
        p[m] = total
    return p[n]


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:euler
class Euler(DefinedFunction):
    """The Euler numbers ``euler(n)`` and polynomials ``euler(n, x)``,
    unevaluated (``euler`` evaluates)."""

    function_name = 'euler'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.numbers import pi
        from sympy_extras.functions.special.gamma_functions import digamma
        n = args[0]
        x: Optional[Expr] = args[1] if len(args) > 1 else None
        if n.is_zero:
            return Integer(1)
        if n == Integer(-1):
            if x is None:
                return div(pi, Integer(2))
            return sub(digamma(div(add(x, Integer(1)), Integer(2))), digamma(div(x, Integer(2))))
        if n.is_integer is False or n.is_nonnegative is False:
            return None
        k = _integer(n)
        if x is None:
            if n.is_odd and n.is_positive:
                return Integer(0)
            if k is not None:
                return Integer(int(mpmath.eulernum(k, exact=True)))
            return None
        if k is not None:
            return _polynomial(_euler_coefficients(k), x)
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        if mpmath.im(n) != 0 or mpmath.re(n) != int(mpmath.re(n)) or mpmath.re(n) < 0:
            return None
        k = int(mpmath.re(n))
        if len(values) == 1:
            return mpmath.mpc(mpmath.eulernum(k))
        return mpmath.mpc(mpmath.eulerpoly(k, values[1]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:bell
class Bell(DefinedFunction):
    """The Bell numbers ``bell(n)`` and polynomials ``bell(n, x)``,
    unevaluated (``bell`` evaluates)."""

    function_name = 'bell'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.numbers import Infinity, oo
        n = args[0]
        x: Optional[Expr] = args[1] if len(args) > 1 else None
        if isinstance(n, Infinity):
            if x is None:
                return oo
            raise ValueError("Bell polynomial is not defined")
        if n.is_negative or n.is_integer is False:
            raise ValueError("a non-negative integer expected")
        k = _integer(n)
        if k is None:
            return None
        if x is None:
            return Integer(sum(_stirling2(k, j) for j in range(k + 1)))
        return add(*[mul(Integer(_stirling2(k, j)), pow(x, Integer(j))) for j in range(k + 1) if _stirling2(k, j)])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if len(values) == 1:
            return mpmath.mpc(mpmath.bell(values[0]))
        return mpmath.mpc(mpmath.bell(values[0], values[1]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:genocchi
class Genocchi(DefinedFunction):
    """The Genocchi numbers ``genocchi(n)`` and polynomials
    ``genocchi(n, x)``, unevaluated (``genocchi`` evaluates)."""

    function_name = 'genocchi'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        n = args[0]
        x: Optional[Expr] = args[1] if len(args) > 1 else None
        if x is not None and x == Integer(1):
            return genocchi(n)
        if n.is_integer is False or n.is_nonnegative is False:
            return None
        k = _integer(n)
        if x is None:
            if n.is_odd and sub(n, Integer(1)).is_positive:
                return Integer(0)
            if k is not None:
                return mul(Integer(2), sub(Integer(1), pow(Integer(2), n)), bernoulli(n))
            return None
        if k is not None:
            return _polynomial(_genocchi_coefficients(k), x)
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        if len(values) != 1 or mpmath.im(n) != 0 or mpmath.re(n) != int(mpmath.re(n)) or mpmath.re(n) < 0:
            return None
        k = int(mpmath.re(n))
        b = mpmath.bernoulli(k) if k != 1 else mpmath.mpf(0.5)
        return mpmath.mpc(2 * (1 - 2**k) * b)

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        if len(self.args) > 1 and self.args[1] != Integer(1):
            return facts
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            facts['integer'] = True
            half = div(n, Integer(2))
            if n.is_odd:
                minus_one_positive = sub(n, Integer(1)).is_positive
                if minus_one_positive is not None:
                    facts['negative'] = not minus_one_positive
                    facts['odd'] = not minus_one_positive
                    facts['even'] = minus_one_positive
                facts['positive'] = False
            else:
                if n.is_zero or n.is_odd:
                    facts['positive'] = False
                elif half.is_even is not None:
                    facts['positive'] = half.is_even
                if half.is_odd is not None:
                    facts['negative'] = half.is_odd
                if n.is_even:
                    if n.is_zero is not None:
                        facts['even'] = n.is_zero
                        facts['odd'] = not n.is_zero
        prime = sub(n, Integer(8)).is_zero
        if prime is not None:
            facts['prime'] = prime
        return facts


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:partition
class Partition(DefinedFunction):
    """The partition numbers ``partition(n)``, unevaluated (``partition``
    evaluates)."""

    function_name = 'partition'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [n] = args
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_negative is True:
            return Integer(0)
        if n.is_zero is True or n == Integer(1):
            return Integer(1)
        k = _integer(n)
        if k is not None:
            return Integer(_partitions(k))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        if mpmath.im(n) != 0 or mpmath.re(n) != int(mpmath.re(n)):
            return None
        return mpmath.mpc(_partitions(int(mpmath.re(n))) if mpmath.re(n) >= 0 else 0)

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        if self.args[0].is_nonnegative is True:
            facts['positive'] = True
        return facts


def euler(n: ExprLike, x: Optional[ExprLike] = None) -> Expr:
    """``euler(n)`` or the polynomial ``euler(n, x)``, evaluated."""
    if x is None:
        return Euler.apply(to_expr(n))
    return Euler.apply(to_expr(n), to_expr(x))


def bell(n: ExprLike, x: Optional[ExprLike] = None) -> Expr:
    """``bell(n)`` or the polynomial ``bell(n, x)``, evaluated."""
    if x is None:
        return Bell.apply(to_expr(n))
    return Bell.apply(to_expr(n), to_expr(x))


def genocchi(n: ExprLike, x: Optional[ExprLike] = None) -> Expr:
    """``genocchi(n)`` or the polynomial ``genocchi(n, x)``, evaluated."""
    if x is None:
        return Genocchi.apply(to_expr(n))
    return Genocchi.apply(to_expr(n), to_expr(x))


def partition(n: ExprLike) -> Expr:
    """``partition(n)``, evaluated."""
    return Partition.apply(to_expr(n))


_INTEGER = {'commutative': True, 'integer': True, 'rational': True, 'real': True, 'extended_real': True,
            'finite': True, 'complex': True, 'algebraic': True}


def _check_positive_integer(n: Expr, name: str = 'n') -> None:
    if n.is_integer is False:
        raise TypeError("%s should be an integer" % name)
    if n.is_positive is False:
        raise ValueError("%s should be a positive integer" % name)


def _factors(n: int) -> dict[int, int]:
    from sympy_extras.ntheory import factorint
    return factorint(n)


class _NumberTheoretic(DefinedFunction):
    """The class-level facts of SymPy's number-theoretic functions."""

    #: the sign facts beyond integer (positive for the totients, ...)
    sign_facts: tuple[tuple[str, bool], ...] = ()

    def _eval_facts(self) -> dict[str, bool]:
        facts = dict(_INTEGER)
        facts.update(self.sign_facts)
        return facts


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:totient
class Totient(_NumberTheoretic):
    """Euler's totient ``totient(n)``, unevaluated (``totient`` evaluates)."""

    function_name = 'totient'
    generic_nseries = True
    sign_facts = (('positive', True),)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [n] = args
        _check_positive_integer(n)
        if n == Integer(1):
            return Integer(1)
        if n.is_prime is True:
            return sub(n, Integer(1))
        k = _integer(n)
        if k is not None:
            product = 1
            for p, e in _factors(k).items():
                product *= p**(e - 1) * (p - 1)
            return Integer(product)
        return None


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:reduced_totient
class ReducedTotient(_NumberTheoretic):
    """Carmichael's reduced totient ``reduced_totient(n)``, unevaluated
    (``reduced_totient`` evaluates)."""

    function_name = 'reduced_totient'
    generic_nseries = True
    sign_facts = (('positive', True),)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from math import lcm
        [n] = args
        _check_positive_integer(n)
        if n == Integer(1):
            return Integer(1)
        if n.is_prime is True:
            return sub(n, Integer(1))
        k = _integer(n)
        if k is None:
            return None
        twos = 0
        while k % 2 == 0:
            k //= 2
            twos += 1
        t = 1 if not twos else (1 << (twos - 2) if twos > 2 else twos)
        return Integer(lcm(t, *[(p - 1) * p**(e - 1) for p, e in _factors(k).items()]))


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:divisor_sigma
class DivisorSigma(_NumberTheoretic):
    """The divisor function ``divisor_sigma(n, k) = sum(d**k, d | n)``,
    unevaluated (``divisor_sigma`` evaluates)."""

    function_name = 'divisor_sigma'
    generic_nseries = True
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})
    sign_facts = (('positive', True),)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        n = args[0]
        k: Expr = args[1] if len(args) > 1 else Integer(1)
        _check_positive_integer(n)
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        if n.is_prime is True:
            return add(Integer(1), pow(n, k))
        if n == Integer(1):
            return Integer(1)
        value = _integer(n)
        if value is None:
            return None
        factors = _factors(value)
        if k.is_zero is True:
            return mul(*[Integer(e + 1) for e in factors.values()])
        order = _integer(k)
        if order is not None:
            product = 1
            for p, e in factors.items():
                product *= (p**(order * (e + 1)) - 1) // (p**order - 1)
            return Integer(product)
        if k.is_zero is False:
            # SymPy: cancel((p**(k*(e + 1)) - 1)/(p**k - 1)) = sum(p**(k*i), i = 0..e)
            return mul(*[add(*[pow(Integer(p), mul(k, Integer(i))) for i in range(e + 1)]) for p, e in factors.items()])
        return None


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:udivisor_sigma
class UDivisorSigma(_NumberTheoretic):
    """The unitary divisor function ``udivisor_sigma(n, k)``, unevaluated
    (``udivisor_sigma`` evaluates)."""

    function_name = 'udivisor_sigma'
    generic_nseries = True
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})
    sign_facts = (('positive', True),)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        n = args[0]
        k: Expr = args[1] if len(args) > 1 else Integer(1)
        _check_positive_integer(n)
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        if n.is_prime is True:
            return add(Integer(1), pow(n, k))
        value = _integer(n)
        if value is None:
            return None
        return mul(*[add(Integer(1), pow(Integer(p), mul(k, Integer(e)))) for p, e in _factors(value).items()])


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:primenu
class PrimeNu(_NumberTheoretic):
    """The number of distinct prime factors ``primenu(n)``, unevaluated
    (``primenu`` evaluates)."""

    function_name = 'primenu'
    generic_nseries = True
    sign_facts = (('nonnegative', True),)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [n] = args
        _check_positive_integer(n)
        if n.is_prime is True:
            return Integer(1)
        if n == Integer(1):
            return Integer(0)
        value = _integer(n)
        return Integer(len(_factors(value))) if value is not None else None


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:primeomega
class PrimeOmega(_NumberTheoretic):
    """The number of prime factors with multiplicity ``primeomega(n)``,
    unevaluated (``primeomega`` evaluates)."""

    function_name = 'primeomega'
    generic_nseries = True
    sign_facts = (('nonnegative', True),)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [n] = args
        _check_positive_integer(n)
        if n.is_prime is True:
            return Integer(1)
        if n == Integer(1):
            return Integer(0)
        value = _integer(n)
        return Integer(sum(_factors(value).values())) if value is not None else None


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:mobius
class Mobius(_NumberTheoretic):
    """The Moebius function ``mobius(n)``, unevaluated (``mobius``
    evaluates)."""

    function_name = 'mobius'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.mul import Mul
        from sympy_extras.core.relational import lt
        from sympy_extras.core.logic import false, true
        [n] = args
        _check_positive_integer(n)
        if n.is_prime is True:
            return Integer(-1)
        if n == Integer(1):
            return Integer(1)
        result: Optional[Expr] = None
        for factor in (n.args if isinstance(n, Mul) else (n,)):
            m, e = factor.as_base_exp()
            if not (m.is_integer is True and m.is_positive is True and e.is_integer is True and e.is_positive is True):
                return None
            relation = lt(Integer(1), e)
            if relation == true:
                result = Integer(0)
                continue
            value = _integer(m)
            if value is not None:
                factors = _factors(value)
                if any(v > 1 for v in factors.values()):
                    result = Integer(0)
                elif relation == false:
                    sign: Expr = Integer(-1) if len(factors) % 2 else Integer(1)
                    result = sign if result is None else mul(result, sign)
        return result


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:primepi
class PrimePi(_NumberTheoretic):
    """The prime counting function ``primepi(n)``, unevaluated
    (``primepi`` evaluates)."""

    function_name = 'primepi'
    sign_facts = (('nonnegative', True),)

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.logic import true
        from sympy_extras.core.numbers import Float, Infinity, NegativeInfinity, oo
        from sympy_extras.core.relational import lt
        from sympy_extras.ntheory import primerange
        [n] = args
        if isinstance(n, Infinity):
            return oo
        if isinstance(n, NegativeInfinity):
            return Integer(0)
        if n.is_real is False:
            raise TypeError("n should be a real")
        if lt(n, Integer(2)) == true:
            return Integer(0)
        if not isinstance(n, (Rational, Float)):
            return None
        bound = int(n.as_fraction())
        return Integer(sum(1 for _ in primerange(2, bound + 1)))


def _legendre(a: int, p: int) -> int:
    import builtins
    value = builtins.pow(a % p, (p - 1) // 2, p)
    return -1 if value == p - 1 else value


def _kronecker(a: int, n: int) -> int:
    """The Kronecker symbol ``(a/n)``, extending Jacobi's to every ``n``."""
    from sympy_extras.ntheory import jacobi_symbol
    if n == 0:
        return 1 if abs(a) == 1 else 0
    sign = 1
    if n < 0:
        n = -n
        if a < 0:
            sign = -1
    twos = 0
    while n % 2 == 0:
        n //= 2
        twos += 1
    if twos:
        if a % 2 == 0:
            return 0
        if twos % 2 and a % 8 in (3, 5):
            sign = -sign
    return sign * (jacobi_symbol(a, n) if n > 1 else 1)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:legendre_symbol
class LegendreSymbol(_NumberTheoretic):
    """The Legendre symbol ``legendre_symbol(a, p)``, unevaluated
    (``legendre_symbol`` evaluates)."""

    function_name = 'legendre_symbol'
    generic_nseries = True
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.mod import mod
        a, p = args
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        if p.is_integer is False:
            raise TypeError("p should be an integer")
        if p.is_prime is False or p.is_odd is False:
            raise ValueError("p should be an odd prime integer")
        if mod(a, p).is_zero is True:
            return Integer(0)
        if a == Integer(1):
            return Integer(1)
        aa, pp = _integer(a), _integer(p)
        if aa is not None and pp is not None:
            return Integer(_legendre(aa, pp))
        return None


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:jacobi_symbol
class JacobiSymbol(_NumberTheoretic):
    """The Jacobi symbol ``jacobi_symbol(m, n)``, unevaluated
    (``jacobi_symbol`` evaluates)."""

    function_name = 'jacobi_symbol'
    generic_nseries = True
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.mod import mod
        from sympy_extras.ntheory import jacobi_symbol as integer_jacobi
        m, n = args
        if m.is_integer is False:
            raise TypeError("m should be an integer")
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False or n.is_odd is False:
            raise ValueError("n should be an odd positive integer")
        if m == Integer(1) or n == Integer(1):
            return Integer(1)
        if mod(m, n).is_zero is True:
            return Integer(0)
        mm, nn = _integer(m), _integer(n)
        if mm is not None and nn is not None:
            return Integer(integer_jacobi(mm, nn))
        return None


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:kronecker_symbol
class KroneckerSymbol(_NumberTheoretic):
    """The Kronecker symbol ``kronecker_symbol(a, n)``, unevaluated
    (``kronecker_symbol`` evaluates)."""

    function_name = 'kronecker_symbol'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        a, n = args
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if a == Integer(1) or n == Integer(1):
            return Integer(1)
        aa, nn = _integer(a), _integer(n)
        if aa is not None and nn is not None:
            return Integer(_kronecker(aa, nn))
        return None


def totient(n: ExprLike) -> Expr:
    """``totient(n)``, evaluated."""
    return Totient.apply(to_expr(n))


def reduced_totient(n: ExprLike) -> Expr:
    """``reduced_totient(n)``, evaluated."""
    return ReducedTotient.apply(to_expr(n))


def divisor_sigma(n: ExprLike, k: Optional[ExprLike] = None) -> Expr:
    """``divisor_sigma(n, k)`` (``k = 1`` by default), evaluated."""
    if k is None:
        return DivisorSigma.apply(to_expr(n))
    return DivisorSigma.apply(to_expr(n), to_expr(k))


def udivisor_sigma(n: ExprLike, k: Optional[ExprLike] = None) -> Expr:
    """``udivisor_sigma(n, k)`` (``k = 1`` by default), evaluated."""
    if k is None:
        return UDivisorSigma.apply(to_expr(n))
    return UDivisorSigma.apply(to_expr(n), to_expr(k))


def primenu(n: ExprLike) -> Expr:
    """``primenu(n)``, evaluated."""
    return PrimeNu.apply(to_expr(n))


def primeomega(n: ExprLike) -> Expr:
    """``primeomega(n)``, evaluated."""
    return PrimeOmega.apply(to_expr(n))


def mobius(n: ExprLike) -> Expr:
    """``mobius(n)``, evaluated."""
    return Mobius.apply(to_expr(n))


def primepi(n: ExprLike) -> Expr:
    """``primepi(n)``, evaluated."""
    return PrimePi.apply(to_expr(n))


def legendre_symbol(a: ExprLike, p: ExprLike) -> Expr:
    """``legendre_symbol(a, p)``, evaluated."""
    return LegendreSymbol.apply(to_expr(a), to_expr(p))


def jacobi_symbol(m: ExprLike, n: ExprLike) -> Expr:
    """``jacobi_symbol(m, n)``, evaluated."""
    return JacobiSymbol.apply(to_expr(m), to_expr(n))


def kronecker_symbol(a: ExprLike, n: ExprLike) -> Expr:
    """``kronecker_symbol(a, n)``, evaluated."""
    return KroneckerSymbol.apply(to_expr(a), to_expr(n))



# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:tribonacci
class Tribonacci(DefinedFunction):
    """The tribonacci numbers ``tribonacci(n)`` and polynomials
    ``tribonacci(n, x)``, unevaluated (``tribonacci`` evaluates)."""

    function_name = 'tribonacci'
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
        if k < 0:
            raise ValueError("Tribonacci polynomials are defined only for non-negative integer indices.")
        if len(args) == 1:
            a, b, c = 0, 1, 1
            for _ in range(k):
                a, b, c = b, c, a + b + c
            return Integer(a)
        x = args[1]
        # T_0 = 0, T_1 = 1, T_2 = x**2, T_k = x**2*T_{k-1} + x*T_{k-2} + T_{k-3}
        polys: list[list[int]] = [[0], [1], [0, 0, 1]]
        for _ in range(3, k + 1):
            p1, p2, p3 = polys[-1], polys[-2], polys[-3]
            size = max(len(p1) + 2, len(p2) + 1, len(p3))
            new = [0] * size
            for i, c1 in enumerate(p1):
                new[i + 2] += c1
            for i, c2 in enumerate(p2):
                new[i + 1] += c2
            for i, c3 in enumerate(p3):
                new[i] += c3
            polys.append(new)
        return add(*[mul(Integer(c), pow(x, Integer(i))) for i, c in enumerate(polys[k]) if c])

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def _motzkin_number(n: int) -> int:
    """The Motzkin number ``M_n`` (``M_0 = M_1 = 1``)."""
    values = [1, 1]
    for k in range(2, n + 1):
        values.append(((2 * k + 1) * values[k - 1] + (3 * k - 3) * values[k - 2]) // (k + 2))
    return values[n]


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:motzkin
class Motzkin(DefinedFunction):
    """SymPy's ``motzkin(n)``: the Motzkin number of ``n - 1`` (``1`` at
    ``n = 0``), as SymPy computes it; unevaluated (``motzkin``
    evaluates)."""

    function_name = 'motzkin'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [n] = args
        k = _integer(n)
        if k is None or k < 0:
            raise ValueError('The provided number must be a positive integer')
        return Integer(_motzkin_number(max(k - 1, 0)))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:andre
class Andre(DefinedFunction):
    """The Andre numbers ``andre(n)`` (the up/down numbers), unevaluated
    (``andre`` evaluates)."""

    function_name = 'andre'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.numbers import Catalan as CatalanConstant, I, Infinity, NaN, nan, number_from_fraction, oo
        from sympy_extras.functions.elementary.complexes import abs_
        from sympy_extras.functions.elementary.exponential import log
        from sympy_extras.functions.special.zeta_functions import zeta
        [n] = args
        if isinstance(n, NaN):
            return nan
        if isinstance(n, Infinity):
            return oo
        if n.is_zero:
            return Integer(1)
        k = _integer(n)
        if k == -1:
            return mul(Integer(-1), log(Integer(2)))
        if k == -2:
            return mul(Integer(-2), CatalanConstant)
        if k is not None:
            if k >= 0 and k % 2 == 0:
                return abs_(euler(n))
            if k % 2:
                m = -k - 1
                ratio = Fraction(1 - Fraction(2)**m) / Fraction(4)**m
                return mul(pow(I, Integer(m)), number_from_fraction(ratio), zeta(mul(Integer(-1), n)))
        return None

    def _eval_facts(self) -> dict[str, bool]:
        n = self.args[0]
        facts: dict[str, bool] = {'commutative': True}
        if n.is_integer and n.is_nonnegative:
            facts['integer'] = True
        if n.is_nonnegative:
            facts['positive'] = True
        return facts


def tribonacci(n: ExprLike, x: Optional[ExprLike] = None) -> Expr:
    """``tribonacci(n)`` or the polynomial ``tribonacci(n, x)``, evaluated."""
    if x is None:
        return Tribonacci.apply(to_expr(n))
    return Tribonacci.apply(to_expr(n), to_expr(x))


def motzkin(n: ExprLike) -> Expr:
    """SymPy's ``motzkin(n)``, evaluated."""
    return Motzkin.apply(to_expr(n))


def andre(n: ExprLike) -> Expr:
    """``andre(n)``, evaluated."""
    return Andre.apply(to_expr(n))
