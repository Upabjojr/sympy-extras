"""Summability methods for divergent oscillatory integrals: the Abel,
Cesàro and Gaussian means of `\\int_a^\\infty f(x)\\, dx` when the
integral does not converge but its regularisations have a limit.

**Abel summability** [Hardy]_ (section 4.3, and 1.16 of [Titchmarsh]_ for
integrals): the value

.. math::

    \\lim_{\\varepsilon \\to 0^+} \\int_a^\\infty f(x)\\, e^{-\\varepsilon x}\\, dx,

so that `\\int_0^\\infty \\sin x\\, dx = 1` and `\\int_0^\\infty \\cos x\\, dx = 0`.
The **Gaussian** mean uses `e^{-\\varepsilon x^2}` instead; it agrees with
the Abel mean whenever both exist, and its inner integral closes for some
integrands whose exponential one does not.

**Cesàro summability** `(C, k)` [Hardy]_ (section 5.14): the value

.. math::

    \\lim_{R \\to \\infty} \\int_a^R \\Bigl(1 - \\frac{x - a}{R - a}\\Bigr)^k f(x)\\, dx,

`k = 1` the mean of the partial integrals `F(X) = \\int_a^X f`,
higher `k` when the lower orders oscillate (`x \\sin x` is `(C, 2)`
summable to `0` but not `(C, 1)`).

The consistency theorems [Hardy]_ (sections 5.12, 5.14): a convergent
integral has every one of these means equal to its value, and Cesàro
summability of some order implies Abel summability to the same value.
Where a mean does not exist (`\\sin^2 x`, whose regularised integral grows
like `1/(4 \\varepsilon)`; `x^k` with `k \\ge 0`) the functions return
``None``, never a finite number.

Examples
========

>>> from sympy import symbols, sin, cos, exp, oo
>>> from sympy_extras.integrals.summability import summable_integral
>>> x = symbols('x')
>>> summable_integral(sin(x), x, 0, oo, 'abel')
ConditionalValue(1)
>>> summable_integral(cos(x), x, 0, oo, 'cesaro')
ConditionalValue(0)
>>> summable_integral(x*cos(x), x, 0, oo, 'abel')
ConditionalValue(-1)
>>> summable_integral(sin(x)**2, x, 0, oo, 'abel') is None
True

References
==========

.. [Hardy] G. H. Hardy, *Divergent Series*, Oxford University Press, 1949,
   chapters IV (Abel and Cesàro means) and V (integrals: sections 5.12,
   5.14, 5.15).
.. [Titchmarsh] E. C. Titchmarsh, *Introduction to the Theory of Fourier
   Integrals*, Oxford University Press, 2nd ed., 1948, sections 1.16-1.17.
"""
from __future__ import annotations

from typing import Optional

from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.expr import Expr
from sympy.core.numbers import nan, oo, zoo
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.logic.boolalg import And, Boolean, true
from sympy.series.limits import Limit

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.limits import limit
from sympy_extras.settings import settings
from .conditions import ConditionalValue, decide

__all__ = ['summable_integral', 'oscillatory_tail', 'METHODS']

#: the summability methods: the Abel mean with the kernel ``exp(-eps*x)``,
#: the Gaussian mean with ``exp(-eps*x**2)``, the Cesàro means ``(C, k)``
#: with the kernels ``(1 - x/R)**k``, ``k = 1, 2, 3``
METHODS = ('abel', 'cesaro', 'gaussian')

#: the Cesàro orders tried in turn
_CESARO_ORDERS = (1, 2, 3)


def summable_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, method: str,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """The ``method`` mean (``'abel'``, ``'cesaro'`` or ``'gaussian'``) of
    ``Integral(f, (x, a, b))`` with the condition on the parameters under
    which it holds, or ``None`` when the mean does not exist or cannot
    be found. A range unbounded on both sides is cut at ``0``, each half
    summed on its own (the means of the two tails must exist separately,
    as Hardy's definitions require); a bounded range is the ordinary
    integral.

    Examples
    ========

    >>> from sympy import symbols, sin, cos, exp, I, oo
    >>> from sympy_extras.integrals.summability import summable_integral
    >>> x = symbols('x')
    >>> a = symbols('a', positive=True)
    >>> summable_integral(sin(a*x), x, 0, oo, 'abel')
    ConditionalValue(1/a)
    >>> summable_integral(exp(I*x), x, 0, oo, 'gaussian')
    ConditionalValue(I)
    >>> summable_integral(cos(x), x, -oo, oo, 'abel')
    ConditionalValue(0)
    >>> summable_integral(x, x, 0, oo, 'abel') is None
    True
    """
    if method not in METHODS:
        raise ValueError("method must be one of %s, got %r" % (", ".join(METHODS), method))
    f_, a_, b_ = as_expr(f), as_expr(a), as_expr(b)
    if a_ == b_:
        return ConditionalValue(S.Zero)
    if a_ == oo or b_ == -oo:
        found = summable_integral(f_, x, b_, a_, method, assumptions)
        return None if found is None else found.scaled(S.NegativeOne)
    if a_ == -oo and b_ == oo:
        right = oscillatory_tail(f_, x, S.Zero, method, assumptions)
        if right is None:
            return None
        left = oscillatory_tail(as_expr(f_.subs(x, -x)), x, S.Zero, method, assumptions)
        return None if left is None else left.add(right)
    if a_ == -oo:
        # x = b - t maps (-oo, b) onto (0, oo)
        t = Dummy('t')
        return oscillatory_tail(as_expr(f_.subs(x, b_ - t)), t, S.Zero, method, assumptions)
    if b_ == oo:
        return oscillatory_tail(f_, x, a_, method, assumptions)
    from .definite import conditional_integral
    return conditional_integral(f_, x, a_, b_, assumptions)


def oscillatory_tail(f: ExprLike, x: Symbol, a: ExprLike, method: str,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """The ``method`` mean of the tail ``Integral(f, (x, a, oo))``, or
    ``None``: the shifted integrand ``f(a + t)`` on ``(0, oo)`` times
    the kernel of the method, integrated by
    :func:`~sympy_extras.integrals.definite_integral` under the
    assumptions and ``eps > 0`` (``R > 0``), and the limit ``eps -> 0+``
    (``R -> oo``) under the assumptions; the mean exists when the limit
    is a finite number free of the regulator. The conditions of the
    inner integral which the assumptions do not settle are kept when
    they do not involve the regulator.

    Examples
    ========

    >>> from sympy import symbols, sin, oo, pi
    >>> from sympy_extras.integrals.summability import oscillatory_tail
    >>> x = symbols('x')
    >>> oscillatory_tail(sin(x), x, pi, 'abel')
    ConditionalValue(-1)
    >>> oscillatory_tail(x*sin(x), x, 0, 'cesaro')
    ConditionalValue(0)
    """
    if method not in METHODS:
        raise ValueError("method must be one of %s, got %r" % (", ".join(METHODS), method))
    f_, a_ = as_expr(f), as_expr(a)
    t = Dummy('t', positive=True)
    g = as_expr(f_.subs(x, t + a_)) if a_ != 0 else as_expr(f_.subs(x, t))
    # the consistency theorem: a convergent integral is its own mean (the
    # partial integrals of a Bessel function, and its Gaussian
    # regularisation, do not close)
    from .definite import conditional_integral
    found = attempt(lambda: conditional_integral(g, t, S.Zero, oo, assumptions), settings.timeout)
    if found is not None and not found.value.has(nan, zoo, oo, -oo, AccumBounds):
        return found
    if method == 'cesaro':
        for order in _CESARO_ORDERS:
            found = _cesaro_mean(g, t, order, assumptions)
            if found is not None:
                return found
        return None
    eps = Dummy('epsilon', positive=True)
    kernel = exp(-eps * t) if method == 'abel' else exp(-eps * t**2)
    return _regularised_limit(as_expr(g * kernel), t, S.Zero, oo, eps, S.Zero, '+', assumptions)


def _cesaro_mean(g: Expr, t: Symbol, order: int, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """The ``(C, order)`` mean of ``Integral(g, (t, 0, oo))``."""
    R = Dummy('R', positive=True)
    kernel = as_expr((1 - t / R)**order)
    return _regularised_limit(as_expr(g * kernel), t, S.Zero, R, R, oo, '-', assumptions)


def _regularised_limit(h: Expr, t: Symbol, lower: Expr, upper: Expr, regulator: Symbol, point: Expr,
                       direction: str, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``Integral(h, (t, lower, upper))`` under the assumptions and
    ``regulator > 0``, then its limit as the regulator tends to
    ``point``; ``None`` unless the limit is a finite number free of the
    regulator."""
    from .definite import conditional_integral
    facts = _items(assumptions) + [as_boolean(regulator > 0)]
    inner = attempt(lambda: conditional_integral(h, t, lower, upper, facts), settings.timeout)
    if inner is None or inner.value.has(nan, zoo, AccumBounds):
        return None
    condition = decide(inner.condition, facts)
    if condition is None or condition.has(regulator):
        return None
    value = attempt(lambda: as_expr(limit(inner.value, regulator, point, direction, facts)), settings.timeout)
    if value is None or value.has(regulator, nan, zoo, oo, -oo, AccumBounds, Limit):
        return None
    if value.is_finite is False:
        return None
    return ConditionalValue(value, _without(condition, regulator))


def _items(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [as_boolean(assumptions)]
    return [as_boolean(item) for item in assumptions]


def _without(condition: Boolean, regulator: Symbol) -> Boolean:
    """The conjuncts of ``condition`` free of the regulator (the others
    hold: the regulator is positive and the limit was taken)."""
    if condition is true:
        return true
    parts = condition.args if isinstance(condition, And) else (condition,)
    return as_boolean(And(*[part for part in parts if not part.has(regulator)]))
