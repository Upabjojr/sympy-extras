"""The zeta functions on the typed core: ``Zeta`` (Riemann and Hurwitz),
``DirichletEta``, ``PolyLog`` (unevaluated nodes) and ``zeta``,
``dirichlet_eta``, ``polylog`` (evaluating).

Ported from SymPy's ``sympy.functions.special.zeta_functions``:
``zeta(s, a)`` at the nonpositive integers (Bernoulli polynomials,
``zeta(0) = -1/2``), at the even positive integers (``zeta(4) =
pi**4/90``), the Hurwitz zeta at a positive integer ``a`` reduced to the
Riemann zeta minus a generalised harmonic number, the pole at ``s = 1``,
``oo``; ``dirichlet_eta`` through ``zeta``; ``polylog(s, z)`` at ``z = 0,
1, -1``, the table of the dilogarithm, ``s = 0, -1``. The Bernoulli
numbers follow SymPy's convention ``B_1 = 1/2``. Also ``lerchphi`` (a
node with its derivatives, as in SymPy), ``riemann_xi`` (through
``zeta``) and the Stieltjes constants ``stieltjes(n, a)``. Not ported:
the polar arguments of ``polylog``, the expansions.

Examples
========

>>> from sympy_extras.core import Symbol, Rational, Integer
>>> from sympy_extras.functions.special.zeta_functions import zeta, dirichlet_eta, polylog, Zeta
>>> x, s = Symbol('x'), Symbol('s')
>>> zeta(0), zeta(-1), zeta(2), zeta(4), zeta(2, 3), zeta(3, 3), zeta(1), zeta(-2, x)
(-1/2, -1/12, pi**2/6, pi**4/90, -5/4 + pi**2/6, -9/8 + zeta(3), zoo, -x**3/3 + x**2/2 - x/6)
>>> dirichlet_eta(1), dirichlet_eta(2), polylog(s, 1), polylog(2, Rational(1, 2)), polylog(0, x), Zeta(2)
(log(2), pi**2/12, zeta(s), -log(2)**2/2 + pi**2/12, x/(1 - x), zeta(2))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py (SymPy 1.14),
# with the Bernoulli numbers and polynomials and the harmonic numbers of
# sympy/functions/combinatorial/numbers.py
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# the polar arguments, lerchphi, stieltjes, riemann_xi not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import comb, factorial as integer_factorial
from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import (Catalan, I, Infinity, Integer, NaN, Rational, nan,
                                       number_from_fraction, pi, zoo)
from sympy_extras.core.power import pow, sqrt
from sympy_extras.functions.elementary.exponential import log

__all__ = ['Zeta', 'DirichletEta', 'PolyLog', 'LerchPhi', 'RiemannXi', 'Stieltjes', 'zeta', 'dirichlet_eta', 'polylog',
           'lerchphi', 'riemann_xi', 'stieltjes', 'bernoulli_number', 'bernoulli_polynomial']

ONE = Integer(1)
ZERO = Integer(0)


@lru_cache(maxsize=None)
def bernoulli_number(n: int) -> Fraction:
    """The Bernoulli number ``B_n`` with SymPy's convention ``B_1 = 1/2``
    (``B_n = B_n(1)``)."""
    numbers = [Fraction(1)]
    for m in range(1, n + 1):
        numbers.append(-sum((comb(m + 1, k)*numbers[k] for k in range(m)), Fraction(0))/(m + 1))
    return -numbers[1] if n == 1 else numbers[n]


def bernoulli_polynomial(n: int, x: Expr) -> Expr:
    """The Bernoulli polynomial ``B_n(x) = sum(binomial(n, k)*b_k*x**(n - k))``
    with ``b_1 = -1/2``, evaluated."""
    terms: list[Expr] = []
    for k in range(n + 1):
        b = -Fraction(1, 2) if k == 1 else bernoulli_number(k)
        if b:
            terms.append(mul(number_from_fraction(comb(n, k)*b), pow(x, Integer(n - k))))
    return add(*terms)


def harmonic_number(n: int, m: int) -> Fraction:
    """The generalised harmonic number ``sum(1/k**m, k = 1..n)``."""
    return sum((Fraction(1)/Fraction(k)**m for k in range(1, n + 1)), Fraction(0))


def _int(e: Expr) -> Optional[int]:
    if isinstance(e, Rational) and e.q == 1:
        return e.p
    return None


def _has_zeta(e: Basic) -> bool:
    stack: list[Basic] = [e]
    while stack:
        node = stack.pop()
        if isinstance(node, Zeta):
            return True
        stack.extend(node.args)
    return False


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:zeta
class Zeta(DefinedFunction):
    """The Riemann zeta ``zeta(s)`` and the Hurwitz zeta ``zeta(s, a)``,
    unevaluated (``zeta`` evaluates)."""

    function_name = 'zeta'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        s = args[0]
        given: Optional[Expr] = args[1] if len(args) > 1 else None
        if given is not None and given == ONE:
            return zeta(s)
        if isinstance(s, NaN) or isinstance(given, NaN):
            return nan
        if s == ONE:
            return zoo
        if isinstance(s, Infinity):
            return ONE
        if isinstance(given, Infinity):
            return ZERO
        a: Expr = ONE if given is None else given
        n = _int(s)
        if n is not None and n <= 0:
            return div(bernoulli_polynomial(1 - n, a), sub(s, ONE))
        if a == ONE:
            if n is not None and n % 2 == 0:
                coefficient = -Fraction(bernoulli_number(n)) / (2*integer_factorial(n))
                return mul(number_from_fraction(coefficient), pow(mul(Integer(2), pi, I), s))
            return None
        k = _int(a)
        if n is not None and k is not None and k > 0:
            return sub(zeta(s), number_from_fraction(harmonic_number(k - 1, n)))
        if k is not None and k <= 0 and (s.is_integer is False or s.is_nonpositive is False):
            return nan
        return None

    def fdiff(self, argindex: int) -> Expr:
        if argindex == 2:
            s, a = self.args
            return mul(Integer(-1), s, zeta(add(s, ONE), a))
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        s = values[0]
        a = values[1] if len(values) > 1 else mpmath.mpc(1)
        if s == 1:
            return None
        try:
            return mpmath.mpc(mpmath.zeta(s, a))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        pole = sub(self.args[0], ONE).is_zero
        if pole is not None:
            facts['finite'] = not pole
        return facts


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:dirichlet_eta
class DirichletEta(DefinedFunction):
    """The Dirichlet eta ``dirichlet_eta(s)`` (and ``dirichlet_eta(s, a)``),
    unevaluated (``dirichlet_eta`` evaluates)."""

    function_name = 'dirichlet_eta'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        s = args[0]
        a: Optional[Expr] = args[1] if len(args) > 1 else None
        if a is not None and a == ONE:
            return dirichlet_eta(s)
        if a is None:
            if s == ONE:
                return log(Integer(2))
            z = zeta(s)
            if not _has_zeta(z):
                return mul(sub(ONE, pow(Integer(2), sub(ONE, s))), z)
            return None
        if s == ONE:
            from .gamma_functions import digamma
            return add(sub(log(Integer(2)), digamma(a)), digamma(div(add(a, ONE), Integer(2))))
        z1, z2 = zeta(s, a), zeta(s, div(add(a, ONE), Integer(2)))
        if not _has_zeta(z1) and not _has_zeta(z2):
            return sub(z1, mul(pow(Integer(2), sub(ONE, s)), z2))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        s = values[0]
        if len(values) == 1:
            return mpmath.mpc(mpmath.altzeta(s))
        a = values[1]
        try:
            return mpmath.mpc(mpmath.zeta(s, a) - 2**(1 - s)*mpmath.zeta(s, (a + 1)/2))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


@lru_cache(maxsize=None)
def _dilogarithm_table() -> dict[Expr, Expr]:
    """SymPy's ``_dilogtable``: ``polylog(2, z)`` at special points."""
    five = sqrt(Integer(5))
    half = Rational(1, 2)
    log2 = log(Integer(2))
    g = div(sub(five, ONE), Integer(2))
    G = div(add(five, ONE), Integer(2))
    catalan = Catalan
    return {
        half: sub(div(pow(pi, Integer(2)), Integer(12)), div(pow(log2, Integer(2)), Integer(2))),
        Integer(2): sub(div(pow(pi, Integer(2)), Integer(4)), mul(I, pi, log2)),
        mul(Integer(-1), g): add(div(pow(pi, Integer(2)), Integer(-15)), div(pow(log(g), Integer(2)), Integer(2))),
        mul(Integer(-1), G): sub(div(pow(pi, Integer(2)), Integer(-10)), pow(log(G), Integer(2))),
        div(sub(Integer(3), five), Integer(2)): sub(div(pow(pi, Integer(2)), Integer(15)), pow(log(g), Integer(2))),
        g: sub(div(pow(pi, Integer(2)), Integer(10)), pow(log(g), Integer(2))),
        I: sub(mul(I, catalan), div(pow(pi, Integer(2)), Integer(48))),
        mul(Integer(-1), I): sub(mul(Integer(-1), I, catalan), div(pow(pi, Integer(2)), Integer(48))),
        sub(ONE, I): add(div(pow(pi, Integer(2)), Integer(16)), mul(Integer(-1), I, catalan),
                         mul(Integer(-1), div(mul(pi, I), Integer(4)), log2)),
        add(ONE, I): add(div(pow(pi, Integer(2)), Integer(16)), mul(I, catalan), mul(div(mul(pi, I), Integer(4)), log2)),
        div(sub(ONE, I), Integer(2)): add(div(pow(log2, Integer(2)), Integer(-8)), div(mul(pi, I, log2), Integer(8)),
                                          div(mul(Integer(5), pow(pi, Integer(2))), Integer(96)), mul(Integer(-1), I,
                                                                                                         catalan)),
    }


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:polylog
class PolyLog(DefinedFunction):
    """The polylogarithm ``polylog(s, z)``, unevaluated (``polylog``
    evaluates)."""

    function_name = 'polylog'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        s, z = args
        if z.is_number:
            if z == ONE:
                return zeta(s)
            if z == Integer(-1):
                return mul(Integer(-1), dirichlet_eta(s))
            if z.is_zero:
                return ZERO
            if s == Integer(2):
                value = _dilogarithm_table().get(z)
                if value is not None:
                    return value
        if z.is_zero:
            return ZERO
        one = z.equals(ONE)
        if one:
            return zeta(s)
        if one is False:
            if s.is_zero:
                return div(z, sub(ONE, z))
            if s == Integer(-1):
                return div(z, pow(sub(ONE, z), Integer(2)))
        # polylog is branched, but not over the unit disk
        from sympy_extras.functions.elementary.complexes import PolarLift, abs_, unpolarify
        from sympy_extras.functions.elementary.exponential import ExpPolar
        from sympy_extras.core.logic import true
        if z.has(ExpPolar, PolarLift) and (one or (abs_(z) <= ONE) is true):
            return polylog(s, unpolarify(z))
        return None

    def fdiff(self, argindex: int) -> Expr:
        s, z = self.args
        if argindex == 2:
            return div(polylog(sub(s, ONE), z), z)
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        s, z = values
        try:
            return mpmath.mpc(mpmath.polylog(s, z))
        except (ValueError, ZeroDivisionError, NotImplementedError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        if facts_of(self.args[1]).get('zero'):
            facts['zero'] = True
        return facts


def zeta(s: ExprLike, a: Optional[ExprLike] = None) -> Expr:
    """``zeta(s)`` or the Hurwitz ``zeta(s, a)``, evaluated."""
    if a is None:
        return Zeta.apply(to_expr(s))
    return Zeta.apply(to_expr(s), to_expr(a))


def dirichlet_eta(s: ExprLike, a: Optional[ExprLike] = None) -> Expr:
    """``dirichlet_eta(s)`` or ``dirichlet_eta(s, a)``, evaluated."""
    if a is None:
        return DirichletEta.apply(to_expr(s))
    return DirichletEta.apply(to_expr(s), to_expr(a))


def polylog(s: ExprLike, z: ExprLike) -> Expr:
    """``polylog(s, z)``, evaluated."""
    return PolyLog.apply(to_expr(s), to_expr(z))





# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:lerchphi
class LerchPhi(DefinedFunction):
    """The Lerch transcendent ``lerchphi(z, s, a)`` (no values of its own,
    as in SymPy)."""

    function_name = 'lerchphi'
    nargs = frozenset({3})
    call_nargs = frozenset({3})

    def fdiff(self, argindex: int) -> Expr:
        z, s, a = self.args
        if argindex == 3:
            return mul(Integer(-1), s, lerchphi(z, add(s, ONE), a))
        if argindex == 1:
            return div(sub(lerchphi(z, sub(s, ONE), a), mul(a, lerchphi(z, s, a))), z)
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.lerchphi(*values))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:riemann_xi
class RiemannXi(DefinedFunction):
    """Riemann's xi function ``riemann_xi(s)``, unevaluated
    (``riemann_xi`` evaluates)."""

    function_name = 'riemann_xi'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .gamma_functions import gamma
        [s] = args
        z = zeta(s)
        if s == ZERO or s == ONE:
            return Rational(1, 2)
        if not isinstance(z, Zeta):
            return div(mul(mul(s, sub(s, ONE)), gamma(div(s, Integer(2))), z), mul(Integer(2), pow(pi, div(s, Integer(2)))))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        s = values[0]
        if s in (0, 1):
            return mpmath.mpc(0.5)
        return mpmath.mpc(s * (s - 1) * mpmath.gamma(s / 2) * mpmath.zeta(s) / (2 * mpmath.pi ** (s / 2)))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:stieltjes
class Stieltjes(DefinedFunction):
    """The Stieltjes constants ``stieltjes(n)`` and ``stieltjes(n, a)``,
    unevaluated (``stieltjes`` evaluates)."""

    function_name = 'stieltjes'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.numbers import EulerGamma, Number
        n = args[0]
        a: Optional[Expr] = args[1] if len(args) > 1 else None
        if a is not None:
            if isinstance(a, NaN):
                return nan
            if isinstance(a, Rational) and a.q == 1 and a.p <= 0:
                return zoo
        default_a = a is None or a == ONE
        if isinstance(n, Number):
            if isinstance(n, NaN):
                return nan
            if n.is_negative:
                return zoo
            if not (isinstance(n, Rational) and n.q == 1):
                return zoo
            if n.is_zero and default_a:
                return EulerGamma
        if n.is_extended_negative:
            return zoo
        if n.is_zero and default_a:
            return EulerGamma
        if n.is_integer is False:
            return zoo
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n = values[0]
        if mpmath.im(n) != 0 or mpmath.re(n) != int(mpmath.re(n)):
            return None
        a = values[1] if len(values) > 1 else mpmath.mpc(1)
        try:
            return mpmath.mpc(mpmath.stieltjes(int(mpmath.re(n)), a))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def lerchphi(z: ExprLike, s: ExprLike, a: ExprLike) -> Expr:
    """``lerchphi(z, s, a)``."""
    return LerchPhi.apply(to_expr(z), to_expr(s), to_expr(a))


def riemann_xi(s: ExprLike) -> Expr:
    """``riemann_xi(s)``, evaluated."""
    return RiemannXi.apply(to_expr(s))


def stieltjes(n: ExprLike, a: Optional[ExprLike] = None) -> Expr:
    """``stieltjes(n)`` or ``stieltjes(n, a)``, evaluated."""
    if a is None:
        return Stieltjes.apply(to_expr(n))
    return Stieltjes.apply(to_expr(n), to_expr(a))
