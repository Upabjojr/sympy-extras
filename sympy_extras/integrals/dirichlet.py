"""Integrals of trigonometric sums over powers of the variable: the
Dirichlet, Frullani and Borwein integrals.

For a trigonometric sum

.. math::

    g(x) = a_0 + \\sum_k \\bigl(a_k \\cos(w_k x) + b_k \\sin(w_k x)\\bigr), \\qquad w_k > 0,

(a product of sines and cosines is written as one by the product
formulas) which vanishes to order `n` at `0`, the integral
`\\int_0^\\infty g(x)\\, x^{-n}\\, dx` converges and is elementary: `n - 1`
integrations by parts, whose boundary terms vanish because `g = O(x^n)`
at `0` and `g` is bounded at infinity, bring it to

.. math::

    \\int_0^\\infty \\frac{g(x)}{x^n}\\, dx = \\frac{1}{(n - 1)!} \\int_0^\\infty \\frac{g^{(n-1)}(x)}{x}\\, dx,

and for the derivative `h = g^{(n-1)} = \\sum_k (A_k \\cos(w_k x) + B_k
\\sin(w_k x))`, a sum without constant term with `h(0) = \\sum_k A_k = 0`,

.. math::

    \\int_0^\\infty \\frac{h(x)}{x}\\, dx = -\\sum_k A_k \\log w_k + \\frac{\\pi}{2} \\sum_k B_k:

Dirichlet's integral `\\int_0^\\infty \\sin(wx)/x\\, dx = \\pi/2` for the
sines ([GR]_ 3.721) and Frullani's theorem for the cosines,
`\\int_0^\\infty (\\cos ax - \\cos bx)/x\\, dx = \\log(b/a)` ([GR]_ 3.784),
the cosine part being `\\sum_k A_k (\\cos(w_k x) - \\cos(w_1 x))/x` when the
`A_k` sum to zero. The condition of vanishing to order `n` is on the
Taylor coefficients of `g`: `a_0 + \\sum_k a_k w_k^m = 0` for even `m < n`
and `\\sum_k b_k w_k^m = 0` for odd `m < n`; for `n = 1` the constant
`a_0` must vanish too, for convergence at infinity (the other terms
converge conditionally). Over the real line the integral is the sum of
the two half-lines, the left one being that of `(-1)^n g(-x)`.

These are the integrals of the sinc products `\\prod_k \\sin(x/(2k+1))/(x/(2k+1))`,
which equal `\\pi/2` up to the factor `1/13` and not from `1/15` on
(Borwein and Borwein [Borwein]_): the value comes out of the
computation, not of a table.

The module works on the objects of the core (:mod:`sympy_extras.core`, the
functions of :mod:`sympy_extras.functions`) with the native ``ask``, the
native ``TR8`` of :mod:`sympy_extras.simplify.fu`, ``simplify`` and
``logcombine``, and imports no SymPy:
:mod:`sympy_extras.integrals.dirichlet_sympy` is its face on SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols, oo
>>> from sympy_extras.functions import sin, cos
>>> from sympy_extras.integrals.dirichlet import dirichlet_integral
>>> x = symbols('x')
>>> dirichlet_integral(sin(x)**2/x**2, x, 0, oo)
ConditionalValue(pi/2)
>>> dirichlet_integral((cos(x) - cos(2*x))/x, x, 0, oo)
ConditionalValue(log(2))
>>> dirichlet_integral(sin(x)*sin(3*x)*sin(5*x)*sin(7*x)/(105*x**4), x, -oo, oo)
ConditionalValue(44*pi/315)
>>> dirichlet_integral(sin(x)**2/x, x, 0, oo) is None
True

References
==========

.. [Borwein] D. Borwein, J. M. Borwein, *Some remarkable properties of
   sinc and related integrals*, The Ramanujan Journal 5 (2001) 73–89.
"""
from __future__ import annotations

from typing import Optional, Union

from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, oo, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.exponential import Exp, log
from sympy_extras.functions.elementary.trigonometric import Cos, Sin, cos, sin
from sympy_extras.polys.polytools import cancel
from sympy_extras.simplify.fu import TR8
from sympy_extras.simplify.simplify import logcombine, simplify

from .conditions import ConditionalValue
from .mellin import monomial

__all__ = ['dirichlet_integral', 'trigonometric_sum', 'TrigonometricKind', 'TrigonometricSum']

#: the kind of a term of a trigonometric sum: the class of its function
TrigonometricKind = Union[type[Sin], type[Cos]]

#: a trigonometric sum: the constant term and ``{(w, kind): coefficient}``
#: for the terms ``coefficient * kind(w * x)``, ``kind`` being ``Sin`` or ``Cos``
TrigonometricSum = tuple[Expr, dict[tuple[Expr, TrigonometricKind], Expr]]


def _as_expr(e: Basic) -> Expr:
    if not isinstance(e, Expr):
        raise TypeError("expected an expression of the core, got %r" % (e,))
    return e


def trigonometric_sum(g: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[TrigonometricSum]:
    """``g`` as a constant plus terms ``c*sin(w*x)`` and ``c*cos(w*x)``
    with ``w > 0``: products and powers of sines and cosines written as
    sums by the product formulas, phases expanded, ``exp(I*w*x)`` as
    cosine and sine; ``None`` when ``g`` is not such a sum or the sign of
    a frequency is unknown.

    >>> from sympy_extras.core import symbols, pi
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.integrals.dirichlet import trigonometric_sum
    >>> x = symbols('x')
    >>> constant, terms = trigonometric_sum(sin(x)**2, x)
    >>> constant, [(w, kind.__name__, c) for (w, kind), c in terms.items()]
    (1/2, [(2, 'Cos', -1/2)])
    >>> [(w, kind.__name__, c) for (w, kind), c in trigonometric_sum(sin(x + pi/4), x)[1].items()]
    [(1, 'Sin', sqrt(2)/2), (1, 'Cos', sqrt(2)/2)]
    """
    g = _as_expr(g)
    if g.has(Exp):
        g = _as_expr(g.rewrite(Cos))
    expanded = g
    for _ in range(8):
        # the product formulas leave squares behind (sin(x)**4 is
        # (1 - cos(2*x))**2/4): again until every term is one function
        expanded = TR8(expanded).expand()
        if all(not isinstance(term.as_independent(x, as_Add=False)[1], (Pow, Mul))
               for term in Add.make_args(expanded)):
            break
    constant: Expr = S.Zero
    terms: dict[tuple[Expr, TrigonometricKind], Expr] = {}

    def add(w: Expr, kind: TrigonometricKind, c: Expr) -> None:
        terms[(w, kind)] = terms.get((w, kind), S.Zero) + c

    for term in Add.make_args(expanded):
        c, function = term.as_independent(x, as_Add=False)
        if function == 1:
            constant = constant + c
            continue
        if not isinstance(function, (Sin, Cos)):
            return None
        argument = function.args[0]
        if argument.is_polynomial(x) is not True or not _of_degree_one(argument, x):
            return None
        w, phase = argument.coeff(x), argument.subs(x, S.Zero)
        if w.has(x) or phase.has(x):
            return None
        if ask(w > 0, assumptions) is not True:
            if ask(w < 0, assumptions) is not True:
                return None
            # cos is even, sin odd
            w, phase = -w, -phase
            if isinstance(function, Sin):
                c = -c
        if phase == 0:
            add(w, Sin if isinstance(function, Sin) else Cos, c)
        elif isinstance(function, Sin):
            add(w, Sin, c * cos(phase))
            add(w, Cos, c * sin(phase))
        else:
            add(w, Cos, c * cos(phase))
            add(w, Sin, -c * sin(phase))
    return constant, {key: value for key, value in terms.items() if value != 0}


def _of_degree_one(p: Expr, x: Symbol) -> bool:
    """Whether the polynomial ``p`` has degree one in ``x`` (its
    derivative is free of ``x`` and not zero), whatever its coefficients
    (SymPy's ``degree`` over ``EX``, which the typed ``Poly`` does not
    have)."""
    d = p.diff(x).expand()
    return not d.has(x) and d != 0


def _vanishes_to_order(total: TrigonometricSum, n: int) -> bool:
    """Whether the sum is ``O(x**n)`` at 0: its Taylor coefficients below
    ``x**n`` vanish identically."""
    constant, terms = total
    for m in range(n):
        coefficient = constant if m == 0 else S.Zero
        for (w, kind), c in terms.items():
            if m % 2 == 0 and kind is Cos:
                coefficient = coefficient + c * w**m
            elif m % 2 == 1 and kind is Sin:
                coefficient = coefficient + c * w**m
        if simplify(coefficient) != 0:
            return False
    return True


def _derivative(total: TrigonometricSum) -> TrigonometricSum:
    """The derivative of the sum in ``x``, a sum without constant term."""
    _, terms = total
    derived: dict[tuple[Expr, TrigonometricKind], Expr] = {}
    for (w, kind), c in terms.items():
        if kind is Sin:
            derived[(w, Cos)] = derived.get((w, Cos), S.Zero) + c * w
        else:
            derived[(w, Sin)] = derived.get((w, Sin), S.Zero) - c * w
    return S.Zero, derived


def _reflected(total: TrigonometricSum, n: int) -> TrigonometricSum:
    """``(-1)**n * g(-x)``: the integrand over the negative half-line
    brought to the positive one."""
    constant, terms = total
    factor = Integer(-1)**n
    reflected = {(w, kind): factor * (-c if kind is Sin else c) for (w, kind), c in terms.items()}
    return factor * constant, reflected


def _half_line(total: TrigonometricSum, n: int) -> Optional[Expr]:
    """``Integral(g(x)/x**n, (x, 0, oo))`` for the sum ``g``, or ``None``
    when it diverges (the sum does not vanish to order ``n`` at 0, or has
    a constant term with ``n = 1``)."""
    if not _vanishes_to_order(total, n):
        return None
    if n == 1 and total[0] != 0:
        return None
    current = total
    for _ in range(n - 1):
        current = _derivative(current)
    value: Expr = S.Zero
    for (w, kind), c in current[1].items():
        value = value + (pi * c / 2 if kind is Sin else -c * log(w))
    # the frequencies are positive: their logarithms combine
    return logcombine(value / factorial(Integer(n - 1)), force=True)


def dirichlet_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` for ``f`` a trigonometric sum (or product)
    over ``x**n`` and the range ``(0, oo)``, ``(-oo, 0)`` or ``(-oo, oo)``;
    ``None`` when ``f`` is not of this shape or the integral diverges.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import sin, cos
    >>> from sympy_extras.integrals.dirichlet import dirichlet_integral
    >>> x = symbols('x')
    >>> dirichlet_integral(sin(x)**3/x**3, x, 0, oo)
    ConditionalValue(3*pi/8)
    >>> dirichlet_integral((1 - cos(x))/x**2, x, -oo, oo)
    ConditionalValue(pi)
    >>> dirichlet_integral(sin(x)/x**2, x, 0, oo) is None
    True
    """
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    if (a_, b_) not in ((S.Zero, oo), (-oo, S.Zero), (-oo, oo)) or not f_.has(Sin, Cos, Exp):
        return None
    numerator, denominator = cancel(f_).as_numer_denom()
    found = monomial(denominator, x)
    if found is None or not isinstance(found[1], Integer) or int(found[1]) < 1:
        return None
    scale, n = found[0], int(found[1])
    total = trigonometric_sum(numerator / scale, x, assumptions)
    if total is None:
        return None
    value: Expr = S.Zero
    if b_ == oo:
        right = _half_line(total, n)
        if right is None:
            return None
        value = value + right
    if a_ == -oo:
        left = _half_line(_reflected(total, n), n)
        if left is None:
            return None
        value = value + left
    # SymPy's expand, whose log hint is on by default
    return ConditionalValue(value.expand(log=True))
