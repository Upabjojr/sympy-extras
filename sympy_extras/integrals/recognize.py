"""Symbolic-numeric recognition of constants: closed forms guessed from
high precision values with integer relation algorithms.

Given a number computed to many digits (a definite integral by
quadrature, a sum, a limit), an **integer relation** algorithm looks for
integers `m_0, m_1, \\ldots, m_n`, not all zero, with

.. math::

    m_0\\, v + m_1 c_1 + \\cdots + m_n c_n = 0

for a chosen basis of constants `c_i` (`\\pi`, `\\log 2`, `\\zeta(3)`,
Catalan's constant, ...). PSLQ [Ferguson]_ finds such a relation, or
proves that none exists with small coefficients at the working
precision; this is the "experimental mathematics" of Bailey and Borwein
[Bailey]_, which discovered many closed forms later proved. mpmath's
``identify`` and ``pslq`` do the work here.

**This is an oracle, not a proof.** A recognised closed form is a
conjecture: it agrees with the number to the digits asked for, and the
candidate is re-checked at a higher precision before it is returned,
but no algebraic derivation supports it. The functions return ``None``
when :data:`sympy_extras.settings.settings.numerical_checks` is off, and
the results must be labelled as conjectures wherever they are used.

The module works on the objects of the core (:mod:`sympy_extras.core`, the
functions of :mod:`sympy_extras.functions`): the integrands are evaluated
by the numerical rules of the core's nodes, and SymPy is not imported.
:mod:`sympy_extras.integrals.recognize_sympy` is its face on SymPy objects.

Examples
========

>>> from sympy_extras.core import Float, symbols
>>> from sympy_extras.integrals.recognize import recognize_constant, recognize_integral
>>> recognize_constant(Float('1.6449340668482264364724151666460251892', 38))
pi**2/6
>>> x = symbols('x')
>>> recognize_integral(1/(x**3 + 1), (x, 0, 1))
log(2)/3 + sqrt(3)*pi/9

References
==========

.. [Ferguson] H. R. P. Ferguson, D. H. Bailey, S. Arno, *Analysis of
   PSLQ, an integer relation finding algorithm*, Mathematics of
   Computation 68 (1999), pp. 351-369.
.. [Bailey] D. H. Bailey, J. M. Borwein, *Experimental Mathematics:
   Examples, Methods and Implications*, Notices of the AMS 52 (2005),
   pp. 502-514.
"""
from __future__ import annotations

from typing import Final, Optional, Sequence, Union

import mpmath

from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Float, Integer, Rational, oo, pi
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.exponential import log
from sympy_extras.functions.special.gamma_functions import gamma
from sympy_extras.functions.special.zeta_functions import zeta
from sympy_extras.settings import settings

__all__ = ['DEFAULT_CONSTANTS', 'recognize_constant', 'recognize_integral', 'quadrature']

#: the constants a closed form is searched in: a value is recognised as a
#: rational combination of them (with 1)
DEFAULT_CONSTANTS: tuple[Expr, ...] = (
    pi, pi**2, pi**3, log(2), log(3), pi * log(2), log(2)**2, sqrt(2), sqrt(3), sqrt(5), sqrt(2) * pi, sqrt(3) * pi,
    S.EulerGamma, S.Catalan, zeta(3), S.Exp1, gamma(Rational(1, 4))**2 / sqrt(pi), gamma(Rational(1, 3))**3,
)

#: the largest coefficient accepted in an integer relation
_MAX_COEFFICIENT = 10**6

#: the guard digits of the numerical values of exact expressions
_GUARD = 20


def _to_mpf(value: Union[Expr, float, str], digits: int) -> Optional[mpmath.mpf]:
    """``value`` as a real mpmath number to ``digits`` digits (an exact
    expression computed with guard digits); ``None`` when it is not a
    real number or cannot be computed."""
    if isinstance(value, (float, str)):
        with mpmath.workdps(digits):
            return mpmath.mpf(value)
    with mpmath.workdps(digits + _GUARD):
        number = numeric_value(value, mpmath.mp.prec)
    if number is None or mpmath.im(number) != 0:
        return None
    with mpmath.workdps(digits):
        return mpmath.mpf(mpmath.re(number))


def recognize_constant(value: Union[Expr, float, str], constants: Sequence[Expr] = DEFAULT_CONSTANTS,
                       digits: int = 30) -> Optional[Expr]:
    """A closed form of ``value`` as a rational combination of 1 and at
    most two of the ``constants``, found by PSLQ at ``digits`` digits;
    ``None`` when no relation with small integer coefficients exists, or
    when numerical checks are off.

    An exact expression is re-evaluated at ``digits + 15`` digits and the
    candidate must agree with it there; a float, a string or a ``Float``
    carries no more digits than it shows, so for those the relation at
    ``digits`` digits is all the evidence (the caller checks further, as
    :func:`recognize_integral` does with a second quadrature).

    The candidate is a conjecture (see the module documentation).

    Examples
    ========

    >>> from sympy_extras.core import Float
    >>> from sympy_extras.integrals.recognize import recognize_constant
    >>> recognize_constant(Float('0.91596559417721901505460351493238411077', 38))
    Catalan
    >>> recognize_constant(Float('1.0986122886681096913952452369225257046', 38))
    log(3)
    >>> recognize_constant(Float('0.123456789012345678901234567890', 30)) is None
    True
    """
    if not settings.numerical_checks:
        return None
    if isinstance(value, Expr) and value.free_symbols:
        return None
    if isinstance(value, Rational):
        return value
    inexact = isinstance(value, (float, str, Float))
    if inexact:
        # the digits the value actually carries
        shown = len(str(value).split('e')[0].replace('.', '').replace('-', '').lstrip('0'))
        digits = min(digits, shown)
        if digits < 12:
            return None
    target = _to_mpf(value, digits)
    if target is None or not mpmath.isfinite(target):
        return None
    candidate = _relation(target, list(constants), digits)
    if candidate is None:
        return None
    if inexact:
        return candidate
    # the check at a higher precision: the value must be recomputable
    # there, otherwise nothing is claimed
    precise = _to_mpf(value, digits + 15)
    guess = _to_mpf(candidate, digits + 15)
    if precise is None or guess is None:
        return None
    with mpmath.workdps(digits + 15):
        if abs(precise - guess) > mpmath.mpf(10)**(-(digits + 10)) * (1 + abs(precise)):
            return None
    return candidate


def _relation(target: mpmath.mpf, constants: list[Expr], digits: int) -> Optional[Expr]:
    """``target`` as a rational combination of 1 and at most two of the
    constants, by PSLQ on the short vectors (a relation among many
    constants needs many more digits than a definite integral is
    computed to: about six digits per constant for coefficients up to a
    million). The pairs are tried after the single constants, and the
    first relation found wins."""
    values: list[mpmath.mpf] = []
    for c in constants:
        value = _to_mpf(c, digits)
        if value is None:
            return None
        values.append(value)
    with mpmath.workdps(digits):
        subsets: list[list[int]] = [[]] + [[i] for i in range(len(constants))]
        subsets += [[i, j] for i in range(len(constants)) for j in range(i + 1, len(constants))]
        for subset in subsets:
            vector = [target, mpmath.mpf(1)] + [values[i] for i in subset]
            try:
                relation = mpmath.pslq(vector, maxcoeff=_MAX_COEFFICIENT, maxsteps=10**4,
                                       tol=mpmath.mpf(10)**(-(digits - 4)))
            except (ValueError, ZeroDivisionError):
                relation = None
            if relation is None or relation[0] == 0:
                continue
            coefficients = [int(m) for m in relation]
            candidate: Expr = -Integer(coefficients[1])
            for m, i in zip(coefficients[2:], subset):
                candidate = candidate - Integer(m) * constants[i]
            return candidate / Integer(coefficients[0])
    return None


class _Integrand:
    """``f`` as a function of the value of ``x``, evaluated by the
    numerical rules of the nodes at the working precision (what
    ``lambdify(x, f, 'mpmath')`` does on SymPy objects); the subtrees free
    of ``x`` are evaluated once per precision. A node with no numerical
    rule raises ``ValueError``."""

    def __init__(self, f: Expr, x: Symbol) -> None:
        self.f: Final[Expr] = f
        self.x: Final[Symbol] = x
        self.constants: Final[dict[tuple[Basic, int], mpmath.mpc]] = {}

    def __call__(self, t: mpmath.mpf) -> mpmath.mpc:
        return self.value(self.f, mpmath.mpc(t))

    def value(self, e: Expr, t: mpmath.mpc) -> mpmath.mpc:
        if e == self.x:
            return t
        if not e.has(self.x):
            key = (e, int(mpmath.mp.prec))
            if key not in self.constants:
                found = numeric_value(e, mpmath.mp.prec)
                if found is None:
                    raise ValueError('%s has no numerical value' % (e,))
                self.constants[key] = found
            return self.constants[key]
        if isinstance(e, Add):
            total = mpmath.mpc(0)
            for a in e.args:
                total = total + self.value(as_expr(a), t)
            return total
        if isinstance(e, Mul):
            product = mpmath.mpc(1)
            for a in e.args:
                product = product * self.value(as_expr(a), t)
            return product
        if isinstance(e, Pow):
            base, exponent = self.value(e.base, t), self.value(e.exp, t)
            if mpmath.im(base) == 0 and mpmath.im(exponent) == 0:
                # real arguments as lambdify's: mpmath.power of mpf values
                return mpmath.mpc(mpmath.power(mpmath.re(base), mpmath.re(exponent)))
            return mpmath.mpc(mpmath.power(base, exponent))
        if isinstance(e, DefinedFunction):
            arguments = [self.value(as_expr(a), t) for a in e.args]
            found = e.numeric(arguments)
            if found is None:
                raise ValueError('%s has no numerical value' % (e,))
            return mpmath.mpc(found)
        raise ValueError('%s cannot be evaluated' % (e,))


def _endpoint(e: Expr, digits: int) -> Optional[mpmath.mpf]:
    if e == oo:
        return mpmath.mpf('inf')
    if e == -oo:
        return mpmath.mpf('-inf')
    return _to_mpf(e, digits)


def quadrature(f: Expr, x: Symbol, a: Expr, b: Expr, digits: int = 30) -> Optional[mpmath.mpf]:
    """``Integral(f, (x, a, b))`` by mpmath's tanh-sinh and Gauss-Legendre
    rules at ``digits`` digits, returned only when the two agree to
    ``digits - 5`` digits and the estimated error is as small; ``None``
    otherwise (a complex or oscillatory integrand, a singularity the
    rules cannot resolve, a node without a numerical rule)."""
    g = _Integrand(f, x)
    lo_, hi_ = _endpoint(a, digits), _endpoint(b, digits)
    if lo_ is None or hi_ is None:
        return None
    with mpmath.workdps(digits):
        lo, hi = mpmath.mpf(lo_), mpmath.mpf(hi_)
        points: list[mpmath.mpf] = [lo, hi]
        if lo == mpmath.mpf('-inf') and hi == mpmath.mpf('inf'):
            points = [lo, mpmath.mpf(-1), mpmath.mpf(0), mpmath.mpf(1), hi]
        elif hi == mpmath.mpf('inf'):
            points = [lo, lo + 1, lo + 10, hi]
        elif lo == mpmath.mpf('-inf'):
            points = [lo, hi - 10, hi - 1, hi]
        try:
            first, error = mpmath.quad(g, points, method='tanh-sinh', error=True)
            second = mpmath.quad(g, points, method='gauss-legendre')
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, mpmath.libmp.NoConvergence):
            return None
        if isinstance(first, mpmath.mpc) or isinstance(second, mpmath.mpc):
            if abs(mpmath.im(first)) > mpmath.mpf(10)**(-(digits - 5)):
                return None
            first, second = mpmath.re(first), mpmath.re(second)
        tolerance = mpmath.mpf(10)**(-(digits - 5)) * (1 + abs(first))
        if abs(first - second) > tolerance or abs(error) > tolerance:
            return None
        return mpmath.mpf(first)


def recognize_integral(f: ExprLike, limits: tuple[Symbol, ExprLike, ExprLike],
                       assumptions: Assumptions = None, constants: Sequence[Expr] = DEFAULT_CONSTANTS,
                       digits: int = 30) -> Optional[Expr]:
    """A conjectured closed form of ``Integral(f, (x, a, b))``: the
    integral is computed numerically to ``digits`` digits (twice, by two
    quadrature rules which must agree) and the value is recognised by
    :func:`recognize_constant`, then checked at ``digits + 15`` digits by
    a third quadrature. ``None`` for an integrand with parameters, when
    the quadrature is not trusted, when nothing is recognised, or when
    numerical checks are off.

    The result is a conjecture, not a theorem: use it to guess, then
    prove (or verify at a higher precision) before relying on it.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import atan, log
    >>> from sympy_extras.integrals.recognize import recognize_integral
    >>> x = symbols('x')
    >>> recognize_integral(atan(x)/x, (x, 0, 1))
    Catalan
    >>> recognize_integral(log(1 + x)/(1 + x**2), (x, 0, 1))
    pi*log(2)/8
    """
    if not settings.numerical_checks:
        return None
    x, a, b = limits[0], to_expr(limits[1]), to_expr(limits[2])
    f_ = to_expr(f)
    if (f_.free_symbols | a.free_symbols | b.free_symbols) - {x}:
        return None
    value = quadrature(f_, x, a, b, digits)
    if value is None:
        return None
    candidate = recognize_constant(Float(mpmath.nstr(value, digits), digits), constants, digits)
    if candidate is None:
        return None
    precise = quadrature(f_, x, a, b, digits + 15)
    guess = _to_mpf(candidate, digits + 15)
    if precise is None or guess is None:
        return None
    with mpmath.workdps(digits + 15):
        if abs(precise - guess) > mpmath.mpf(10)**(-(digits + 8)) * (1 + abs(precise)):
            return None
    return candidate
