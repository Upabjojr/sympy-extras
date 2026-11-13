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

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native limits under assumptions and the native
conditions. The regularised integrals are computed by the native
:func:`~sympy_extras.integrals.definite.conditional_integral`. The
functions on SymPy objects are in :mod:`.summability_sympy`.

Examples
========

>>> from sympy_extras.core import symbols, oo
>>> from sympy_extras.functions import sin, cos
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

import itertools
from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.limits import limit
from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.limits import Limit
from sympy_extras.core.logic import And, Boolean, BooleanTrue, and_, to_boolean, true
from sympy_extras.core.numbers import nan, oo, zoo
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.settings import settings

from .conditions import ConditionalValue, decide

__all__ = ['summable_integral', 'oscillatory_tail', 'METHODS']

#: the summability methods: the Abel mean with the kernel ``exp(-eps*x)``,
#: the Gaussian mean with ``exp(-eps*x**2)``, the Cesàro means ``(C, k)``
#: with the kernels ``(1 - x/R)**k``, ``k = 1, 2, 3``
METHODS: Final = ('abel', 'cesaro', 'gaussian')

#: the Cesàro orders tried in turn
_CESARO_ORDERS: Final = (1, 2, 3)

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so a dummy ``t`` beside a parameter ``t`` would be
    taken for it."""
    return Dummy('%s%d' % (name, next(_COUNTER)), **assumptions)


def _integral(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions,
              seconds: Optional[float]) -> Optional[ConditionalValue]:
    """:func:`~sympy_extras.integrals.definite.conditional_integral` within
    ``seconds``."""
    from .definite import conditional_integral
    return attempt(lambda: conditional_integral(f, x, a, b, assumptions), seconds, native=True)


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

    >>> from sympy_extras.core import I, Symbol, oo, symbols
    >>> from sympy_extras.functions import sin, cos, exp
    >>> from sympy_extras.integrals.summability import summable_integral
    >>> x = symbols('x')
    >>> a = Symbol('a', positive=True)
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
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    if a_ == b_:
        return ConditionalValue(S.Zero)
    if a_ == oo or b_ == -oo:
        found = summable_integral(f_, x, b_, a_, method, assumptions)
        return None if found is None else found.scaled(S.NegativeOne)
    if a_ == -oo and b_ == oo:
        right = oscillatory_tail(f_, x, S.Zero, method, assumptions)
        if right is None:
            return None
        left = oscillatory_tail(f_.subs(x, -x), x, S.Zero, method, assumptions)
        return None if left is None else left.add(right)
    if a_ == -oo:
        # x = b - t maps (-oo, b) onto (0, oo)
        t = _fresh('t')
        return oscillatory_tail(f_.subs(x, b_ - t), t, S.Zero, method, assumptions)
    if b_ == oo:
        return oscillatory_tail(f_, x, a_, method, assumptions)
    return _integral(f_, x, a_, b_, assumptions, None)


def oscillatory_tail(f: ExprLike, x: Symbol, a: ExprLike, method: str,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """The ``method`` mean of the tail ``Integral(f, (x, a, oo))``, or
    ``None``: the shifted integrand ``f(a + t)`` on ``(0, oo)`` times
    the kernel of the method, integrated by
    :func:`~sympy_extras.integrals.definite.definite_integral` under the
    assumptions and ``eps > 0`` (``R > 0``), and the limit ``eps -> 0+``
    (``R -> oo``) under the assumptions; the mean exists when the limit
    is a finite number free of the regulator. The conditions of the
    inner integral which the assumptions do not settle are kept when
    they do not involve the regulator.

    Examples
    ========

    >>> from sympy_extras.core import symbols, pi
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.integrals.summability import oscillatory_tail
    >>> x = symbols('x')
    >>> oscillatory_tail(sin(x), x, pi, 'abel')
    ConditionalValue(-1)
    >>> oscillatory_tail(x*sin(x), x, 0, 'cesaro')
    ConditionalValue(0)
    """
    if method not in METHODS:
        raise ValueError("method must be one of %s, got %r" % (", ".join(METHODS), method))
    f_, a_ = to_expr(f), to_expr(a)
    t = _fresh('t', positive=True)
    g = f_.subs(x, t + a_) if a_ != 0 else f_.subs(x, t)
    # the consistency theorem: a convergent integral is its own mean (the
    # partial integrals of a Bessel function, and its Gaussian
    # regularisation, do not close)
    found = _integral(g, t, S.Zero, oo, assumptions, settings.timeout)
    if found is not None and not found.value.has(nan, zoo, oo, -oo, AccumBounds):
        return found
    if method == 'cesaro':
        for order in _CESARO_ORDERS:
            found = _cesaro_mean(g, t, order, assumptions)
            if found is not None:
                return found
        return None
    eps = _fresh('epsilon', positive=True)
    kernel = exp(-eps * t) if method == 'abel' else exp(-eps * t**2)
    return _regularised_limit(g * kernel, t, S.Zero, oo, eps, S.Zero, '+', assumptions)


def _cesaro_mean(g: Expr, t: Symbol, order: int, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """The ``(C, order)`` mean of ``Integral(g, (t, 0, oo))``."""
    R = _fresh('R', positive=True)
    kernel = (1 - t / R)**order
    return _regularised_limit(g * kernel, t, S.Zero, R, R, oo, '-', assumptions)


def _regularised_limit(h: Expr, t: Symbol, lower: Expr, upper: Expr, regulator: Symbol, point: Expr,
                       direction: str, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``Integral(h, (t, lower, upper))`` under the assumptions and
    ``regulator > 0``, then its limit as the regulator tends to
    ``point``; ``None`` unless the limit is a finite number free of the
    regulator."""
    facts = _items(assumptions) + [regulator > 0]
    inner = _integral(h, t, lower, upper, facts, settings.timeout)
    if inner is None or inner.value.has(nan, zoo, AccumBounds):
        return None
    condition = decide(inner.condition, facts)
    if condition is None or condition.has(regulator):
        return None
    value = attempt(lambda: limit(inner.value, regulator, point, direction, facts), settings.timeout, native=True)
    if value is None or value.has(regulator, nan, zoo, oo, -oo, AccumBounds, Limit):
        return None
    if value.is_finite is False:
        return None
    return ConditionalValue(value, _without(condition, regulator))


def _items(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Basic, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(item) for item in assumptions]


def _without(condition: Boolean, regulator: Symbol) -> Boolean:
    """The conjuncts of ``condition`` free of the regulator (the others
    hold: the regulator is positive and the limit was taken)."""
    if isinstance(condition, BooleanTrue):
        return true
    parts = condition.args if isinstance(condition, And) else (condition,)
    return and_(*[to_boolean(part) for part in parts if not part.has(regulator)])
