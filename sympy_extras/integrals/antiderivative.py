"""Definite integrals from an antiderivative, with the singularities of
both the integrand and the antiderivative taken into account.

The fundamental theorem of calculus, ``Integral(f, (x, a, b)) = F(b) -
F(a)``, holds when ``F`` is an antiderivative of ``f`` *continuous on
the closed range*. Computer algebra antiderivatives are not: the
logarithms and inverse tangents of a Risch or Risch–Norman antiderivative
jump where their arguments cross a branch cut (``atan(tan(x))`` at
``x = pi/2``), so evaluating at the endpoints gives a wrong answer, and
an integrand singular inside the range gives a finite wrong number where
the integral diverges (``integrate(1/x**2, (x, -1, 1))`` is ``-2`` in
older systems). Rioboo's construction [Rioboo]_ of a continuous real
antiderivative of a rational function repairs the ``atan`` part (SymPy's
``ratint`` uses it); Jeffrey and Rich [Jeffrey]_ do the same for the
trigonometric substitutions. For a definite integral there is a general
repair which needs neither: cut the range at every point where ``F`` or
``f`` is not continuous and add the one-sided limits,

.. math::

    \\int_a^b f\\, dx = \\sum_i \\left( \\lim_{x \\to c_{i+1}^-} F(x) - \\lim_{x \\to c_i^+} F(x) \\right),

with `a = c_0 < c_1 < \\ldots < c_n = b` the singular points. A jump of
``F`` at an interior point is subtracted by the two limits which meet
there, and an infinite limit at a singularity of ``f`` exposes the
divergence instead of hiding it. The limits are taken by
:func:`sympy_extras.assumptions.limits.limit` under the assumptions on
the parameters, the points are found by the native
:func:`sympy_extras.assumptions.solve.solve` and placed by the native
:func:`~sympy_extras.assumptions.ask.ask`.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy. The antiderivative itself is a step of its own, an
:data:`AntiderivativeStep` which every function takes: by default
:func:`antiderivative`, which tries the native table of quadratic
radicals (:mod:`.radicals`) and asks the rest of SymPy's ``integrate``
and of the verified methods of :mod:`.indefinite` (not ported yet)
through the face (:func:`~.antiderivative_sympy.integrate_of_core`,
:func:`~.antiderivative_sympy.verified_antiderivative_of_core`), which
loads SymPy on that path only. A native antiderivative (the native
``ratint`` of :mod:`.risch.rationaltools`, say) runs the whole method
without SymPy. The singularities of the integrand are those of SymPy's
``singularities`` (a port, :func:`singularities`). The functions on SymPy
objects are in :mod:`.antiderivative_sympy`.

Examples
========

>>> from sympy_extras.core import Symbol, S, oo, pi
>>> from sympy_extras.functions import cos
>>> from sympy_extras.integrals.antiderivative import antiderivative_integral
>>> x = Symbol('x')
>>> antiderivative_integral(1/(2 + cos(x)), x, S.Zero, 2*pi)
ConditionalValue(2*sqrt(3)*pi/3)
>>> antiderivative_integral(1/x**2, x, -S.One, S.One)
ConditionalValue(oo)

References
==========

.. [Rioboo] R. Rioboo, *Quelques aspects du calcul exact avec des nombres
   réels*, thèse, Université Paris 6, 1991; and M. Bronstein, *Symbolic
   Integration I*, 2nd ed., Springer, 2005, section 2.8 (Rioboo's
   algorithm for real rational function integration).
.. [Jeffrey] D. J. Jeffrey, A. D. Rich, *The evaluation of trigonometric
   integrals avoiding spurious discontinuities*, ACM Transactions on
   Mathematical Software 20 (1994) 124–135.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/calculus/singularities.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, the points as a list (None when a set is not
# finite) solved by the native solve under the assumptions on the range.
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import random
from typing import Callable, Final, Optional

from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.limits import limit
from sympy_extras.assumptions.solve import solve
from sympy_extras.core.accumbounds import AccumBounds
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import Application, expand
from sympy_extras.core.limits import Limit
from sympy_extras.core.logic import Boolean, to_boolean, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, Rational, nan, oo, pi, zoo
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Relational
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs, Im, Sign, abs_, im
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log
from sympy_extras.functions.elementary.hyperbolic import (Acoth, Acsch, Asech, Atanh, Cosh, Coth, Csch, Sech, Sinh,
                                                          Tanh, cosh, sinh)
from sympy_extras.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy_extras.functions.elementary.trigonometric import Acot, Atan, Cos, Cot, Csc, Sec, Sin, Tan, cos, sin
from sympy_extras.functions.special.delta_functions import Heaviside
from sympy_extras.functions.special.error_functions import Erf, erf
from sympy_extras.polys.polytools import cancel
from sympy_extras.sets import EmptySet, FiniteSet, Reals
from sympy_extras.settings import settings
from sympy_extras.simplify.powsimp import powsimp

from .conditions import ConditionalValue, sample_values

__all__ = ['AntiderivativeStep', 'antiderivative', 'elementary_polynomial', 'quick_shape', 'by_parts_shape',
           'singularities', 'discontinuities', 'antiderivative_integral', 'one_sided_limit', 'select_branch',
           'real_logarithms', 'principal_value_integral', 'finite_part_integral']


#: an antiderivative of ``f`` in ``x`` under the assumptions, by the exact
#: methods or (with the last argument ``True``) by the heuristic ones;
#: ``None`` when none is found. :func:`antiderivative` is the default
AntiderivativeStep = Callable[[Expr, Symbol, Assumptions, bool], Optional[Expr]]


#: the methods of the indefinite dispatcher tried for a definite integral:
#: the exact ones, which fail fast; the heuristics (the Risch–Norman
#: method, the substitutions, SymPy's manual and Meijer routes) spend
#: their whole budget on every piece of every mapped range
_METHODS: Final[tuple[str, ...]] = ('rational', 'radicals', 'exponential', 'trigonometric', 'risch', 'trager',
                                    'binomial', 'sympy')


#: the heuristic methods, tried late and under a larger budget
_LATE_METHODS: Final[tuple[str, ...]] = ('heurisch', 'rewriting', 'rewritten heurisch', 'manual', 'meijer', 'sympy')

#: the classes of the elementary functions of linear arguments which
#: integrate takes at once, with their evaluating functions
_ELEMENTARY: Final[dict[type[Basic], Callable[[ExprLike], Expr]]] = {
    Sin: sin, Cos: cos, Exp: exp, Sinh: sinh, Cosh: cosh}

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so a dummy ``x`` beside the variable ``x`` would be
    taken for it."""
    return Dummy('%s_%d' % (name, next(_COUNTER)), **assumptions)


def _budget(share: int) -> Optional[float]:
    return None if settings.timeout is None else settings.timeout / share


def antiderivative(f: Expr, x: Symbol, assumptions: Assumptions = None, late: bool = False) -> Optional[Expr]:
    """An antiderivative of ``f`` by the radical table, by ``integrate``
    for a polynomial in elementary functions, and by the verified methods
    of :mod:`.indefinite` otherwise, under an eighth of the time limit;
    ``None`` when none is found (an unevaluated ``Integral`` in the result
    counts as none). The assumptions on the parameters reach the checks
    (``1/(cosh(n*t)**2 + 1)`` has an antiderivative for ``n > 0``, none
    that checks for a general ``n``). With ``late`` the heuristic methods
    are tried instead, under a quarter of the time limit (the second pass
    of the definite driver, after its slow methods).

    The radical table is native; SymPy's ``integrate`` and the methods of
    :mod:`.indefinite` (not ported) are reached through
    :mod:`.antiderivative_sympy`, which loads SymPy.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.integrals.antiderivative import antiderivative
    >>> x = Symbol('x')
    >>> antiderivative(1/sqrt(2*x - x**2), x)
    asin(x - 1)
    """
    from .radicals import quadratic_radical_antiderivative
    found = quadratic_radical_antiderivative(f, x)
    if found is not None:
        # a real form (SymPy writes the arcsine of sqrt(1 - x**2) as a
        # complex logarithm, whose limits at algebraic bounds fail)
        return found
    from .antiderivative_sympy import integrate_of_core, verified_antiderivative_of_core
    if elementary_polynomial(f, x):
        # a polynomial in x and in exponentials and trigonometric functions
        # of linear arguments: integrate does it at once, where the Risch
        # port spends seconds on gcds over the constants of the arguments
        # (sin(pi*t/4 + pi/4)**3 took 7 s, on every quarter period); the
        # arguments expanded, since sin(pi*(t/4 + 1/4)) takes integrate
        # 8 s where sin(pi*t/4 + pi/4) takes a tenth of one
        g = as_expr(f.replace(lambda n: type(n) in _ELEMENTARY, _expanded_argument))
        found = integrate_of_core(g, x, settings.timeout)
        if found is not None:
            return found
    found = _logarithm_by_parts(f, x)
    if found is not None:
        return found
    found = _error_function_by_parts(f, x)
    if found is not None:
        return found
    # the exact typed methods of indefinite integration in their order
    # (the tables, the trigonometric integrator, the Risch port, Trager's
    # algorithm, SymPy's integrate last), every candidate checked by
    # differentiation:
    # 1/(cosh(n*t)**2 + 1) has a real logarithmic form there where the
    # Risch port gives up and integrate answers a Piecewise in tanh;
    # an eighth of the time limit: the methods may spend it all, the
    # route is tried on every piece of every mapped range, and the other
    # methods of the driver need their share (log(sin(x)/x) over
    # (0, pi/2) lost its budget to the failures on its pieces)
    return verified_antiderivative_of_core(f, x, assumptions, list(_LATE_METHODS if late else _METHODS),
                                           _budget(4 if late else 8))


def _expanded_argument(node: Basic) -> Basic:
    """An elementary function of :data:`_ELEMENTARY` with its argument
    expanded."""
    function = _ELEMENTARY[type(node)]
    return function(expand(as_expr(node.args[0])))


def elementary_polynomial(f: Expr, x: Symbol) -> bool:
    """Whether ``f`` is a polynomial in ``x`` and in sines, cosines,
    exponentials and hyperbolic functions of arguments linear in ``x``.

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.functions import sin, exp, log
    >>> from sympy_extras.integrals.antiderivative import elementary_polynomial
    >>> t = Symbol('t')
    >>> elementary_polynomial(sin(pi*t/4 + pi/4)**3*exp(2*t), t), elementary_polynomial(log(t)*sin(t), t)
    (True, False)
    """
    replacement: dict[Basic, Basic] = {}
    for node in f.atoms(Application):
        if not node.has(x):
            continue
        if type(node) not in _ELEMENTARY or len(node.args) != 1:
            return False
        argument = as_expr(node.args[0])
        if not _linear(argument, x):
            return False
        replacement[node] = _fresh('e')
    substituted = as_expr(f._xreplace_evaluated(replacement))
    variables = [x] + [s for s in replacement.values() if isinstance(s, Symbol)]
    return substituted.is_polynomial(*variables) is True


def _linear(argument: Expr, x: Symbol) -> bool:
    """Whether ``argument`` is a polynomial of degree one in ``x`` (its
    derivative a nonzero constant: the native ``degree`` takes no
    coefficients outside the ported domains, ``pi/4``)."""
    if argument.is_polynomial(x) is not True:
        return False
    slope = argument.diff(x)
    return not slope.has(x) and slope.is_zero is not True


def quick_shape(f: Expr, x: Symbol) -> bool:
    """Whether ``f`` is of a shape the antiderivative comes at once for: a
    polynomial in ``x`` and in elementary functions of linear arguments,
    or such a polynomial with exponentials times ``log(x)`` or times one
    error function.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp, log, sin
    >>> from sympy_extras.integrals.antiderivative import quick_shape
    >>> x = Symbol('x')
    >>> quick_shape(x**3*exp(-x)*log(x), x), quick_shape(sin(x)/x, x)
    (True, False)
    """
    return elementary_polynomial(f, x) or by_parts_shape(f, x) or _error_function_shape(f, x)


def _error_functions(f: Expr) -> list[Expr]:
    return [factor for factor in Mul.make_args(f) if isinstance(factor, Erf)]


def _error_function_shape(f: Expr, x: Symbol) -> bool:
    errors = _error_functions(f)
    return len(errors) == 1 and elementary_polynomial(f / errors[0], x)


def by_parts_shape(f: Expr, x: Symbol) -> bool:
    """Whether ``f`` is a polynomial in ``x`` and exponentials times ``log(x)``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.integrals.antiderivative import by_parts_shape
    >>> x = Symbol('x')
    >>> by_parts_shape(x**3*exp(-x)*log(x), x), by_parts_shape(x*log(x), x)
    (True, False)
    """
    logarithms = [factor for factor in Mul.make_args(f) if isinstance(factor, Log)]
    if len(logarithms) != 1 or logarithms[0] != log(x):
        return False
    g = f / log(x)
    return not g.has(Log) and elementary_polynomial(g, x) and g.has(Exp)


def _error_function_by_parts(f: Expr, x: Symbol) -> Optional[Expr]:
    """``g(x)*erf(k*x)`` for a polynomial ``g`` in ``x`` and in elementary
    functions of linear arguments, by parts: ``G*erf(k*x) -
    2*k/sqrt(pi)*Integral(G*exp(-k**2*x**2))``, the last integrated term
    by term with the exponentials combined (``exp(-w**2 - w)`` completes
    its square at once, where ``integrate`` spent a minute on
    ``cosh(u - w)*exp(-w**2)``): ``sinh(u - w)*erf(w)`` over ``(0, u)``
    (Maxima's ``laplace`` 57). ``None`` for another shape."""
    errors = _error_functions(f)
    if len(errors) != 1:
        return None
    argument = as_expr(errors[0].args[0])
    k, variable = argument.as_independent(x, as_Add=False)
    if variable != x or k.has(x):
        return None
    g = f / errors[0]
    if g.has(Erf) or not elementary_polynomial(g, x):
        return None
    from .antiderivative_sympy import integrate_of_core
    plain = _fresh(x.name)
    g = g.subs(x, plain)
    budget = _budget(8)
    G = integrate_of_core(g, plain, budget)
    if G is None:
        return None
    remainder = expand(G.rewrite(Exp) * exp(-k**2 * plain**2))
    H: Expr = S.Zero
    for term in Add.make_args(remainder):
        piece = integrate_of_core(powsimp(term, combine='exp'), plain, budget)
        if piece is None:
            return None
        H = H + piece
    return (G * erf(k * plain) - 2 * k / sqrt(pi) * H).subs(plain, x)


def _logarithm_by_parts(f: Expr, x: Symbol) -> Optional[Expr]:
    """``g(x)*log(x)`` for a polynomial ``g`` in ``x`` and in exponentials
    of linear arguments, by parts: ``G*log(x) - Integral(G/x)`` with ``G``
    the antiderivative of ``g``, the last an exponential integral
    (``u**3*exp(-u)*log(u)`` in two seconds, where ``integrate`` took
    six). ``None`` for another shape."""
    if not by_parts_shape(f, x):
        return None
    from .antiderivative_sympy import integrate_of_core
    # on a plain dummy: integrate is four times slower on a symbol
    # declared positive (the assumptions are consulted at every step)
    plain = _fresh(x.name)
    g = (f / log(x)).subs(x, plain)
    budget = _budget(8)
    G = integrate_of_core(g, plain, budget)
    if G is None:
        return None
    H = integrate_of_core(cancel(G / plain), plain, budget)
    if H is None:
        return None
    return (G * log(plain) - H).subs(plain, x)


def _not_real(p: Expr, assumptions: Assumptions) -> bool:
    """Whether ``p`` is not real once the parameters the assumptions make
    real are taken as such: ``(log(3 - 2*sqrt(2)) + I*pi)/(2*n)`` for
    ``n > 0`` (a singularity of an antiderivative in ``exp(2*n*t)`` which
    lies off the real line, where SymPy's ``singularities`` leaves it
    intersected with the range)."""
    replacement: dict[Basic, Basic] = {}
    for q in p.free_symbols:
        if not isinstance(q, Symbol):
            continue
        if ask(q > 0, assumptions) is True:
            replacement[q] = _fresh(q.name, positive=True)
        elif ask(q < 0, assumptions) is True:
            replacement[q] = _fresh(q.name, negative=True)
        elif q.is_extended_real or ask(element(q, Reals), assumptions):
            replacement[q] = _fresh(q.name, real=True)
    if not replacement:
        return False
    substituted = as_expr(p._xreplace_evaluated(replacement))
    if substituted.is_extended_real is False:
        return True
    imaginary = im(substituted)
    return not imaginary.has(Im) and imaginary.is_zero is False


def _points_in(points: list[Expr], a: Expr, b: Expr, assumptions: Assumptions) -> Optional[list[Expr]]:
    """The points strictly inside ``(a, b)``, sorted; ``None`` when a
    position is undecided."""
    inside: list[Expr] = []
    for p in points:
        if p.is_extended_real is False or _not_real(p, assumptions):
            continue
        below = True if a == -oo else ask(p > a, assumptions)
        above = True if b == oo else ask(p < b, assumptions)
        if below is None or above is None:
            return None
        if below and above and p not in inside:
            inside.append(p)
    for i in range(len(inside)):
        for j in range(i + 1, len(inside)):
            if ask(inside[i] < inside[j], assumptions) is None:
                return None
    return sorted(inside, key=lambda p: sum(1 for q in inside if ask(q < p, assumptions)))


def _real_zeros(u: Expr, x: Symbol, facts: list[Boolean]) -> Optional[list[Expr]]:
    """The real zeros of ``u`` in ``x`` under the facts by the native
    solve, none when ``u`` has a sign under them; ``None`` when they are
    not a finite set or the time is up."""
    while isinstance(u, Abs):
        # the zeros of the absolute value are those of its argument (the
        # rational argument stays with the native analysis)
        u = as_expr(u.args[0])
    if ask(u > 0, facts) is True or ask(u < 0, facts) is True:
        # no zero on the range (exp(2*n*t) - 2*sqrt(2) + 3 for n > 0,
        # where the solver answers a ConditionSet; the absolute value of
        # sin(x)/tan(1) - cos(x) on (0, 1), an unsimplified intersection
        # of image sets)
        return []
    zeros = attempt(lambda: solve(u, x, facts, domain=Reals), settings.timeout, native=True)
    if zeros is None:
        return None
    if zeros is EmptySet:
        return []
    if not isinstance(zeros, FiniteSet):
        return None
    return [as_expr(p) for p in zeros.args]


# PORTED FROM SYMPY: sympy/calculus/singularities.py:singularities
def singularities(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[list[Expr]]:
    """The real isolated singularities of ``f`` in ``x`` as SymPy's
    ``singularities`` finds them, under the assumptions (``x > a`` and ``x
    < b`` for a range): the zeros of the bases of the negative powers
    once the trigonometric and hyperbolic functions are written in
    cosines, of the arguments of ``log``, ``asech``, ``acsch``, and the
    points ``1`` and ``-1`` of the arguments of ``atanh`` and ``acoth``;
    ``None`` when a set of zeros is not finite or not found.

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.functions import log, tan
    >>> from sympy_extras.integrals.antiderivative import singularities
    >>> x = Symbol('x')
    >>> singularities(1/(x**2 - 1) + log(x + 2), x)
    [-1, 1, -2]
    """
    e = f.rewrite(Sec, Csc, Cot, Tan, Cos)
    e = e.rewrite(Sech, Csch, Coth, Tanh, Cosh)
    targets: list[Expr] = []
    for node in e.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        if node.exp.is_infinite:
            return None
        if node.exp.is_negative:
            # XXX (SymPy): an exponent of varying sign is not handled
            targets.append(node.base)
    for node in f.atoms(Log, Asech, Acsch):
        targets.append(as_expr(node.args[0]))
    for node in f.atoms(Atanh, Acoth):
        targets.append(as_expr(node.args[0]) - 1)
        targets.append(as_expr(node.args[0]) + 1)
    facts = _facts(assumptions)
    points: list[Expr] = []
    for u in targets:
        if not u.has(x):
            continue
        zeros = _real_zeros(u, x, facts)
        if zeros is None:
            return None
        points.extend(p for p in zeros if p not in points)
    return points


def discontinuities(F: Expr, x: Symbol, a: Expr, b: Expr,
                    assumptions: Assumptions = None) -> Optional[list[Expr]]:
    """The points of ``(a, b)`` where the antiderivative ``F`` may be
    discontinuous: its singularities (poles, zeros of the arguments of
    logarithms) and the poles of the arguments of ``atan`` and ``acot``
    (where ``atan`` jumps by ``pi``), together with the kinks of
    ``Abs``, ``sign``, ``Heaviside`` and ``Piecewise``; ``None`` when
    they cannot be located.

    >>> from sympy_extras.core import Symbol, Integer, S
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.antiderivative import discontinuities
    >>> x = Symbol('x')
    >>> discontinuities(log(x - 1) + 1/(x - 2), x, S.Zero, Integer(3))
    [1, 2]
    """
    bounded = _bounded(assumptions, x, a, b)
    candidates = singularities(F, x, bounded)
    if candidates is None:
        return None
    arguments: list[Expr] = []
    for node in F.atoms(Atan, Acot):
        if node.has(x):
            _, denominator = as_expr(node.args[0]).as_numer_denom()
            if denominator.has(x):
                arguments.append(denominator)
    for node in F.atoms(Log):
        if node.has(x):
            arguments.append(as_expr(node.args[0]))
    for node in F.atoms(Abs, Sign, Heaviside):
        if node.has(x):
            arguments.append(as_expr(node.args[0]))
    for node in F.atoms(Piecewise):
        for pair in node.args:
            if not isinstance(pair, ExprCondPair):
                continue
            for relation in pair.cond.atoms(Relational):
                if isinstance(relation, Relational):
                    difference = relation.lhs - relation.rhs
                    if difference.has(x):
                        arguments.append(difference)
    for u in arguments:
        zeros = _real_zeros(u, x, bounded)
        if zeros is None:
            return None
        candidates.extend(zeros)
    return _points_in(candidates, a, b, assumptions)


def _facts(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Basic, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(s) for s in assumptions]


def _bounded(assumptions: Assumptions, x: Symbol, a: Expr, b: Expr) -> list[Boolean]:
    facts: list[Boolean] = []
    if a != -oo:
        facts.append(x > a)
    if b != oo:
        facts.append(x < b)
    return facts + _facts(assumptions)


def select_branch(F: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions = None) -> Optional[Expr]:
    """The branch of a ``Piecewise`` antiderivative which holds on
    ``(a, b)``, decided at a sample point of the range under the
    assumptions (the conditions of SymPy's antiderivatives are on the
    variable, ``x > 1``); ``None`` when a condition cannot be decided or
    the branches change inside the range (the bug: ``limit`` of a
    ``Piecewise`` with an infinite value on the branch of the range gave
    0, and the area of a disc came out as 0).

    >>> from sympy_extras.core import Symbol, Integer, S
    >>> from sympy_extras.functions import Piecewise
    >>> from sympy_extras.integrals.antiderivative import select_branch
    >>> t = Symbol('t')
    >>> F = Piecewise((t**2, t > 1), (t**3, True))
    >>> select_branch(F, t, Integer(2), Integer(3)), select_branch(F, t, S.Zero, Integer(2))
    (t**2, None)
    """
    if a == -oo and b == oo:
        sample: Expr = S.Zero
    elif a == -oo:
        sample = b - 1
    elif b == oo:
        sample = a + 1
    else:
        sample = (a + b) / 2
    result = F
    for node in result.atoms(Piecewise):
        chosen: Optional[Expr] = None
        for pair in node.args:
            if not isinstance(pair, ExprCondPair):
                return None
            value, condition = pair.expr, pair.cond
            if condition is true:
                chosen = value
                break
            # the same branch must hold on the whole range: no condition
            # met on the way may change inside it
            for atom in condition.atoms(Relational):
                if not isinstance(atom, Relational):
                    continue
                zeros = attempt(lambda: solve(atom.lhs - atom.rhs, x, _bounded(assumptions, x, a, b), domain=Reals),
                                settings.timeout, native=True)
                if zeros is None or zeros is not EmptySet:
                    return None
            holds = ask(to_boolean(condition.subs(x, sample)), assumptions)
            if holds is None:
                return None
            if holds:
                chosen = value
                break
        if chosen is None:
            return None
        result = as_expr(result._xreplace_evaluated({node: chosen}))
    return result


def one_sided_limit(F: Expr, x: Symbol, point: Expr, direction: str,
                    assumptions: Assumptions = None) -> Optional[Expr]:
    """``lim F(x)`` as ``x`` tends to ``point`` from the side ``'+'`` or
    ``'-'`` (``F(point)`` when ``F`` is continuous there); ``oo`` or
    ``-oo`` for a signed infinity, and ``None`` when the limit does not
    exist (an oscillation, ``AccumBounds``), is complex infinite, or
    cannot be found.

    >>> from sympy_extras.core import Symbol, S, oo
    >>> from sympy_extras.functions import atan, sin
    >>> from sympy_extras.integrals.antiderivative import one_sided_limit
    >>> x = Symbol('x')
    >>> one_sided_limit(atan(x), x, oo, '-'), one_sided_limit(-1/x, x, S.Zero, '+')
    (pi/2, -oo)
    >>> one_sided_limit(sin(x), x, oo, '-') is None
    True
    """
    if point == oo or point == -oo:
        value = attempt(lambda: limit(F, x, point, assumptions=assumptions), settings.timeout, native=True)
    else:
        value = attempt(lambda: limit(F, x, point, direction, assumptions=assumptions), settings.timeout,
                        native=True)
        if (value is None or value.has(Limit)) and not point.is_number:
            # a symbolic point, where limit gives up: the value of F there
            # when it is finite (the bug: the arcsine of a real antiderivative
            # had no limit at the algebraic bound -sqrt(1 - x**2) of a cell)
            value = F.subs(x, point)
    if value is None:
        return None
    if value == oo or value == -oo:
        return value
    if value.has(oo, -oo, zoo, nan, Limit, Piecewise, AccumBounds):
        return None
    if value.free_symbols - F.free_symbols - point.free_symbols:
        # a dummy of an unevaluated inner limit leaked (the symbols of a
        # symbolic point are its own: the bug refused every limit at the
        # algebraic bound -sqrt(1 - x**2) of a cell)
        return None
    return value


def _prepared(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions, late: bool,
              known: Optional[Expr], step: Optional[AntiderivativeStep]) -> Optional[Expr]:
    """The antiderivative (``known``, or by the ``step``) on its branch of
    the range; ``None`` when there is none or the branch is undecided."""
    if known is not None:
        F: Optional[Expr] = known
    elif step is not None:
        F = step(f, x, assumptions, late)
    else:
        F = antiderivative(f, x, assumptions, late)
    if F is None:
        return None
    if F.has(Piecewise):
        return select_branch(F, x, a, b, assumptions)
    return F


def _cut_points(f: Expr, F: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[list[Expr]]:
    points = discontinuities(F, x, a, b, assumptions)
    if points is None:
        return None
    singular = singularities(f, x, _bounded(assumptions, x, a, b))
    if singular is None:
        return None
    return _points_in(singular + points, a, b, assumptions)


def antiderivative_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                            assumptions: Assumptions = None, late: bool = False,
                            known: Optional[Expr] = None,
                            step: Optional[AntiderivativeStep] = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` from an antiderivative ``F``, cut at the
    discontinuities of ``F`` and the singularities of ``f`` inside the
    range and evaluated by one-sided limits; ``oo`` or ``-oo`` when
    the limits are infinite of one sign (the integral diverges to it, as
    ``1/x**2`` over ``(-1, 1)``); ``None`` when there is no
    antiderivative, a point cannot be placed, a limit is unknown, or the
    infinities cancel (``1/x`` over ``(-1, 1)``, where the principal
    value is another question). An antiderivative already ``known`` is
    used instead of computing one (the cases of the parameters of the
    definite driver share it); otherwise the ``step`` computes it
    (:func:`antiderivative` by default).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, S, oo, pi
    >>> from sympy_extras.functions import cos
    >>> from sympy_extras.integrals.antiderivative import antiderivative_integral
    >>> x = Symbol('x')
    >>> antiderivative_integral(1/(5 - 4*cos(x)), x, S.Zero, 2*pi)
    ConditionalValue(2*pi/3)
    >>> antiderivative_integral(1/(1 + x**2), x, -oo, oo)
    ConditionalValue(pi)
    >>> antiderivative_integral(1/x, x, S.Zero, S.One)
    ConditionalValue(oo)

    A native antiderivative keeps SymPy out (the rational integrator of
    :mod:`.risch.rationaltools`):

    >>> from sympy_extras.integrals.risch.rationaltools import ratint
    >>> antiderivative_integral(1/(x**2 - 4), x, S.Zero, S.One, step=lambda f, x, _, __: ratint(f, x))
    ConditionalValue(-log(3)/4)
    """
    a_, b_ = to_expr(a), to_expr(b)
    prepared = _prepared(f, x, a_, b_, assumptions, late, known, step)
    if prepared is None:
        return None
    F = real_logarithms(prepared, x)
    more = _cut_points(f, F, x, a_, b_, assumptions)
    if more is None:
        return None
    bounds = [a_] + more + [b_]
    total: Expr = S.Zero
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        right = one_sided_limit(F, x, upper, '-', assumptions)
        left = one_sided_limit(F, x, lower, '+', assumptions)
        if right is None or left is None:
            return None
        # an infinite limit is checked against the sign of the integrand
        # near the point (SymPy's limit of x**2*Shi(x)/2 - x*cosh(x)/2 +
        # sinh(x)/2 at oo is -oo; the function grows like x*exp(x)/4)
        if (right == oo or right == -oo) and not _integrand_has_sign(f, x, upper, '-', right, assumptions):
            return None
        if (left == oo or left == -oo) and not _integrand_has_sign(f, x, lower, '+', -left, assumptions):
            return None
        total = total + (right - left)
    if total.has(nan, zoo):
        # oo - oo: infinities of both signs, nothing is claimed
        return None
    return ConditionalValue(_absorbed(total))


def real_logarithms(F: Expr, x: Symbol) -> Expr:
    """``log(u)`` written ``log(Abs(u))`` for the real arguments ``u``
    depending on ``x``: the same antiderivative where ``u`` keeps its
    sign, real where ``u`` is negative (``log(sin(x)/tan(1) - cos(x))``
    on ``(0, 1)`` is ``log(-u) + I*pi``, whose infinite limit at 1 the
    complex values make ``zoo``).

    >>> from sympy_extras.core import Symbol, I
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.antiderivative import real_logarithms
    >>> x = Symbol('x')
    >>> real_logarithms(log(x - 1) + log(x + I), x)
    log(x + I) + log(Abs(x - 1))
    """
    real = _fresh(x.name, real=True)

    def query(node: Basic) -> bool:
        if not isinstance(node, Log) or not node.has(x) or isinstance(node.args[0], Abs):
            return False
        return as_expr(node.args[0]).subs(x, real).is_extended_real is True

    def value(node: Basic) -> Basic:
        return log(abs_(as_expr(node.args[0])))

    return as_expr(F.replace(query, value))


def _absorbed(total: Expr) -> Expr:
    """``total`` with the finite numbers of a sum absorbed into its
    infinity: SymPy keeps ``-Si(1)/2 + oo`` as a sum (the finiteness of
    ``Si(1)`` is not known to it), which is ``oo`` (``sin(x)/x**3`` over
    ``(0, 1)``)."""
    if total == oo or total == -oo or not total.has(oo, -oo):
        return total
    infinite = [term for term in Add.make_args(total) if term.has(oo, -oo)]
    finite = [term for term in Add.make_args(total) if not term.has(oo, -oo)]
    if not finite or not all(term.is_number for term in finite):
        return total
    if all(term == oo for term in infinite):
        return oo
    if all(term == -oo for term in infinite):
        return -oo
    return total


def _integrand_has_sign(f: Expr, x: Symbol, point: Expr, side: str, infinity: Expr,
                        assumptions: Assumptions) -> bool:
    """Whether ``f`` has the sign of ``infinity`` (``oo`` or ``-oo``) at
    three points approaching ``point`` from ``side``, the parameters
    sampled under the assumptions: the sign of the integrand near a
    point where its integral is claimed to diverge to that infinity.
    ``True`` when nothing can be checked (no sample of the parameters, a
    complex value)."""
    parameters = sorted((f.free_symbols | point.free_symbols) - {x}, key=lambda s: s.name)
    symbols = [s for s in parameters if isinstance(s, Symbol)]
    values = sample_values(symbols, assumptions, random.Random(str((f, point)))) if symbols else {}
    if values is None:
        return True
    rule: dict[Basic, Basic] = dict(values.items())
    g, p = as_expr(f._xreplace_evaluated(rule)), as_expr(point._xreplace_evaluated(rule))
    if p == oo:
        samples: list[Expr] = [Integer(10), Integer(100), Integer(1000)]
    elif p == -oo:
        samples = [Integer(-10), Integer(-100), Integer(-1000)]
    else:
        step = 1 if side == '+' else -1
        samples = [p + step * Rational(1, 10**k) for k in (2, 4, 6)]
    positive = infinity == oo
    for sample in samples:
        try:
            value = reliable_value(g.subs(x, sample), 15)
        except ZeroDivisionError:
            return True
        if value is None or value.is_extended_real is not True:
            return True
        if (value.is_extended_negative if positive else value.is_extended_positive) is True:
            return False
    return True


def _regularized(f: Expr, x: Symbol, a: ExprLike, b: ExprLike, assumptions: Assumptions,
                 step: Optional[AntiderivativeStep],
                 jump: Callable[[Expr, Symbol, Expr, Symbol], Optional[Expr]]) -> Optional[ConditionalValue]:
    """The limits at the ends of the range minus the ``jump`` of the
    antiderivative across every point inside, for the principal value and
    the finite part."""
    a_, b_ = to_expr(a), to_expr(b)
    F = _prepared(f, x, a_, b_, assumptions, False, None, step)
    if F is None:
        return None
    inside = _cut_points(f, F, x, a_, b_, assumptions)
    if inside is None:
        return None
    right = one_sided_limit(F, x, b_, '-', assumptions)
    left = one_sided_limit(F, x, a_, '+', assumptions)
    if right is None or left is None or right in (oo, -oo) or left in (oo, -oo):
        # the principal value and the finite part take finite limits at
        # the endpoints only
        return None
    total: Expr = right - left
    epsilon = _fresh('epsilon', positive=True)
    for c in inside:
        excised = F.subs(x, c + epsilon) - F.subs(x, c - epsilon)
        value = jump(excised, epsilon, F, x)
        if value is None or value.free_symbols - F.free_symbols:
            return None
        total = total - value
    return ConditionalValue(total)


def principal_value_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                             assumptions: Assumptions = None,
                             step: Optional[AntiderivativeStep] = None) -> Optional[ConditionalValue]:
    """Cauchy's principal value of ``Integral(f, (x, a, b))`` from an
    antiderivative ``F`` (by the ``step``, :func:`antiderivative` by
    default): with `c_1, \\ldots, c_n` the singularities of ``f`` and the
    discontinuities of ``F`` inside the range,

    .. math::

        \\mathrm{PV}\\int_a^b f\\, dx = \\lim_{x \\to b^-} F - \\lim_{x \\to a^+} F
        - \\sum_i \\lim_{\\epsilon \\to 0^+} \\left( F(c_i + \\epsilon) - F(c_i - \\epsilon) \\right),

    the symmetric limits at each point taken as one limit, in which the
    divergent parts of a simple pole cancel (``log(epsilon)`` for
    ``1/x``) while a stronger singularity leaves an infinite limit, and
    ``None`` is returned. For a convergent integral the value is the
    integral itself.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer, S
    >>> from sympy_extras.integrals.antiderivative import principal_value_integral
    >>> x = Symbol('x')
    >>> principal_value_integral(1/x, x, -S.One, Integer(2))
    ConditionalValue(log(2))
    >>> principal_value_integral(1/x**2, x, -S.One, S.One) is None
    True
    """
    def jump(excised: Expr, epsilon: Symbol, F: Expr, _: Symbol) -> Optional[Expr]:
        value = attempt(lambda: limit(excised, epsilon, 0, assumptions=assumptions), settings.timeout,
                        native=True)
        if value is None or value.has(oo, -oo, zoo, nan, Limit):
            return None
        return value

    return _regularized(f, x, a, b, assumptions, step, jump)


def _finite_part(jump: Expr, epsilon: Symbol) -> Optional[Expr]:
    """The constant term of the expansion of ``jump`` at ``epsilon = 0``,
    the terms in negative powers of ``epsilon`` and in ``log(epsilon)``
    dropped (Hadamard's finite part)."""
    expansion = attempt(lambda: jump.series(epsilon, 0, 1).removeO(), settings.timeout, native=True)
    if expansion is None:
        return None
    constant: Expr = S.Zero
    for term in Add.make_args(expansion.expand()):
        if not term.has(epsilon):
            constant = constant + term
    if constant.has(nan, zoo, oo, -oo):
        return None
    return constant


def finite_part_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                         assumptions: Assumptions = None,
                         step: Optional[AntiderivativeStep] = None) -> Optional[ConditionalValue]:
    """Hadamard's finite part of ``Integral(f, (x, a, b))``: as in
    :func:`principal_value_integral`, but the divergent terms of the
    symmetric excision at each singularity (``1/epsilon``,
    ``log(epsilon)``) are dropped instead of required to cancel, which
    regularises a double pole (``Integral(1/x**2, (x, -1, 1))`` has the
    finite part ``-2``) [Hadamard]_.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer, S
    >>> from sympy_extras.integrals.antiderivative import finite_part_integral
    >>> x = Symbol('x')
    >>> finite_part_integral(1/x**2, x, -S.One, S.One)
    ConditionalValue(-2)
    >>> finite_part_integral(1/x, x, -S.One, Integer(2))
    ConditionalValue(log(2))

    References
    ==========

    .. [Hadamard] J. Hadamard, *Lectures on Cauchy's problem in linear
       partial differential equations*, Yale University Press, 1923,
       chapter on the "partie finie"; the modern account in Estrada and
       Kanwal, *Singular Integral Equations*, Birkhäuser, 2000, chapter 2.
    """
    def jump(excised: Expr, epsilon: Symbol, F: Expr, _: Symbol) -> Optional[Expr]:
        return _finite_part(excised, epsilon)

    return _regularized(f, x, a, b, assumptions, step, jump)
