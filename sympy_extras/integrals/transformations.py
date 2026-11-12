"""Integral identities which change the integrand rather than compute
it: Frullani's theorem, the Cauchy–Schlömilch transformation and
Glasser's master theorem.

**Frullani's theorem** [Frullani]_: for a function with finite limits at
`0` and `\\infty`,

.. math::

    \\int_0^\\infty \\frac{f(ax) - f(bx)}{x}\\, dx = \\bigl(f(0) - f(\\infty)\\bigr) \\log\\frac{b}{a},
    \\qquad a, b > 0.

**Glasser's master theorem** [Glasser]_: for `a_i > 0`,

.. math::

    \\int_{-\\infty}^\\infty F\\left(x - \\sum_i \\frac{a_i}{x - b_i}\\right) dx
    = \\int_{-\\infty}^\\infty F(x)\\, dx,

whenever the right side converges; the **Cauchy–Schlömilch
transformation** is the case of one term and an even `F`,
`\\int_0^\\infty F\\bigl((x - a/x)^2\\bigr) dx = \\int_0^\\infty F(x^2)\\, dx`
[Amdeberhan]_. The theorem holds because the map `x \\mapsto x - \\sum a_i/(x - b_i)`
is a bijection of each interval between consecutive poles onto the real
line with Jacobians summing to one (Boros and Moll, chapter 13).

The functions here recognise the shapes and hand the transformed integral
to :func:`sympy_extras.integrals.definite.definite_integral`.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native limits under assumptions, ask and partial
fractions. The transformed integral is computed by the native
:func:`~sympy_extras.integrals.definite.conditional_integral`. The
functions on SymPy objects are in :mod:`.transformations_sympy`.

Examples
========

>>> from sympy_extras.core import Symbol, oo
>>> from sympy_extras.functions import exp
>>> from sympy_extras.integrals.transformations import frullani, glasser
>>> x = Symbol('x')
>>> a, b = Symbol('a', positive=True), Symbol('b', positive=True)
>>> frullani((exp(-a*x) - exp(-b*x))/x, x, a > 0)
ConditionalValue(log(b/a))
>>> glasser(exp(-(x - 1/x)**2), x, 0, oo)
ConditionalValue(sqrt(pi)/2)

References
==========

.. [Frullani] G. Frullani, *Sopra gli integrali definiti*, Memorie della
   Società Italiana delle Scienze 20 (1828); the modern statement in
   G. Boros, V. Moll, *Irresistible Integrals*, Cambridge University Press,
   2004, section 5.2.
.. [Glasser] M. L. Glasser, *A remarkable property of definite integrals*,
   Mathematics of Computation 40 (1983) 561–563.
.. [Amdeberhan] T. Amdeberhan, M. L. Glasser, M. C. Jones, V. Moll,
   R. Posey, D. Varela, *The Cauchy–Schlömilch transformation*, Integral
   Transforms and Special Functions 30 (2019); Boros and Moll, chapter 13.
"""
from __future__ import annotations

import itertools
from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.limits import limit
from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Application
from sympy_extras.core.limits import Limit
from sympy_extras.core.logic import and_
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, nan, oo, zoo
from sympy_extras.core.power import Pow
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import log
from sympy_extras.polys.partfrac import apart
from sympy_extras.settings import settings

from .conditions import ConditionalValue
from .mellin import monomial

__all__ = ['frullani', 'glasser', 'transformation_integral']

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so a dummy ``y`` beside a parameter ``y`` would be
    taken for it."""
    return Dummy('%s%d' % (name, next(_COUNTER)), **assumptions)


def _expr(e: Basic) -> Expr:
    if not isinstance(e, Expr):
        raise TypeError("expected an expression, got %s" % (e,))
    return e


def _limit(f: Expr, x: Symbol, point: Expr, assumptions: Assumptions) -> Optional[Expr]:
    value = attempt(lambda: limit(f, x, point, assumptions=assumptions), settings.timeout, native=True)
    if value is None or value.has(oo, -oo, zoo, nan, Limit, AccumBounds) or value.free_symbols - f.free_symbols:
        return None
    return value


def frullani(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, 0, oo))`` by Frullani's theorem for
    ``f = c*(g(a*x) - g(b*x))/x``; ``None`` when ``f`` is not of this shape
    or the limits of ``g`` at ``0`` and ``oo`` are not finite. The
    condition ``a > 0`` and ``b > 0`` is reported.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import atan
    >>> from sympy_extras.integrals.transformations import frullani
    >>> x, a, b = Symbol('x'), Symbol('a', positive=True), Symbol('b', positive=True)
    >>> frullani((atan(a*x) - atan(b*x))/x, x)
    ConditionalValue(-pi*log(b/a)/2)
    """
    terms = Add.make_args(f.expand())
    if len(terms) != 2:
        return None
    reciprocal = x**-1
    parts: list[tuple[Expr, Expr, Expr]] = []
    for term in terms:
        coefficient, rest = term.as_independent(x, as_Add=False)
        if not isinstance(rest, Mul):
            return None
        factors = rest.args
        if len(factors) != 2 or reciprocal not in factors:
            return None
        function = factors[0] if factors[1] == reciprocal else factors[1]
        if not isinstance(function, Application) or len(function.args) != 1:
            return None
        found = monomial(_expr(function.args[0]), x)
        if found is None or found[1] != 1:
            return None
        parts.append((coefficient, found[0], _expr(function._evaluated((x,)))))
    (c1, a, g1), (c2, b, g2) = parts
    if g1 != g2 or c1 + c2 != 0:
        return None
    if a.is_negative and b.is_negative:
        # g(a x) = h(|a| x) with h(x) = g(-x)
        a, b, g1 = -a, -b, g1.subs(x, -x)
    if ask(a > 0, assumptions) is False or ask(b > 0, assumptions) is False:
        return None
    at_zero = _limit(g1, x, S.Zero, assumptions)
    at_infinity = _limit(g1, x, oo, assumptions)
    if at_zero is None or at_infinity is None:
        return None
    return ConditionalValue(c1 * (at_zero - at_infinity) * log(b / a), and_(a > 0, b > 0))


def _glasser_map(u: Expr, x: Symbol, assumptions: Assumptions) -> Optional[list[tuple[Expr, Expr]]]:
    """``[(a_i, b_i)]`` when ``u == x - sum(a_i/(x - b_i))`` with ``a_i > 0``."""
    # (not a rational function of x, exp(x) among the generators: apart
    # raises PolynomialError, and attempt gives None)
    decomposed = attempt(lambda: apart(u, x), settings.timeout, native=True)
    if decomposed is None:
        return None
    poles: list[tuple[Expr, Expr]] = []
    linear: Expr = S.Zero
    for term in Add.make_args(decomposed):
        if term == x or not term.has(x):
            linear = linear + term
            continue
        coefficient, rest = term.as_independent(x, as_Add=False)
        if not (isinstance(rest, Pow) and rest.exp == -1):
            return None
        base = rest.base
        # the base linear in x: a slope free of x
        slope = base.diff(x)
        if slope == 0 or slope.has(x):
            return None
        intercept = (base - slope * x).expand()
        if intercept.has(x):
            return None
        a_i = -coefficient / slope
        b_i = -intercept / slope
        if ask(a_i > 0, assumptions) is not True:
            return None
        poles.append((a_i, b_i))
    if linear - x != 0 or not poles:
        return None
    return poles


def _operations(e: Expr) -> int:
    count = e.count_ops()
    return int(count) if isinstance(count, Integer) else 0


def _inner_arguments(f: Expr, x: Symbol) -> list[Expr]:
    """The subexpressions of ``f`` depending on ``x`` which are arguments
    of functions, or bases of powers, largest first (ties in the order
    of their printed forms)."""
    found: set[Expr] = set()
    for node in f.atoms(Application):
        for argument in node.args:
            if isinstance(argument, Expr) and argument.has(x) and argument != x:
                found.add(argument)
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base = node.base
        if base.has(x) and base != x:
            found.add(base)
    return sorted(found, key=lambda e: (-_operations(e), str(e)))


def glasser(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
            assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` by Glasser's master theorem, for ``f =
    F(u(x))`` with ``u = x - sum(a_i/(x - b_i))``, ``a_i > 0``, over the
    real line, or over ``(0, oo)`` when ``F`` is even (the
    Cauchy–Schlömilch transformation, ``f = G(u**2)``); the transformed
    integral ``Integral(F, (x, -oo, oo))`` is computed by
    :func:`~sympy_extras.integrals.definite.conditional_integral`.
    ``None`` when the shape is not recognised.

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.transformations import glasser
    >>> x = Symbol('x')
    >>> glasser(exp(-(x + 1/x)**2), x, -oo, oo) is None
    True
    """
    a_, b_ = to_expr(a), to_expr(b)
    whole_line = (a_, b_) == (-oo, oo)
    half_line = (a_, b_) == (S.Zero, oo)
    if not whole_line and not half_line:
        return None
    y = _fresh('y', real=True)
    for v in _inner_arguments(f, x):
        poles = _glasser_map(v, x, assumptions)
        if poles is None:
            continue
        replaced = _expr(f._xreplace_evaluated({v: y}))
        if replaced.has(x):
            continue
        if half_line and _expr(replaced._xreplace_evaluated({y: -y})) != replaced:
            # over the half line F must be even
            continue
        lower = -oo if whole_line else S.Zero
        from .definite import conditional_integral
        value = conditional_integral(replaced, y, lower, oo, assumptions)
        if value is not None:
            return value
    return None


def transformation_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                            assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """The first of the identities of this module which applies.

    >>> from sympy_extras.core import Symbol, S
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.transformations import transformation_integral
    >>> x = Symbol('x')
    >>> transformation_integral(exp(-x), x, S.Zero, S.One) is None
    True
    """
    a_, b_ = to_expr(a), to_expr(b)
    if (a_, b_) == (S.Zero, oo):
        found = frullani(f, x, assumptions)
        if found is not None:
            return found
    return glasser(f, x, a_, b_, assumptions)
