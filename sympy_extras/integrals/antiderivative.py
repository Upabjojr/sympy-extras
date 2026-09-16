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
divergence instead of hiding it. This module implements that repair
around SymPy's indefinite ``integrate`` (and the Risch port of
:mod:`sympy_extras.integrals.risch` when it applies), with the limits
taken by :func:`sympy_extras.assumptions.limit` under the assumptions on
the parameters.

Examples
========

>>> from sympy import symbols, integrate, cos, pi
>>> from sympy_extras.integrals.antiderivative import antiderivative_integral
>>> x = symbols('x')
>>> antiderivative_integral(1/(2 + cos(x)), x, 0, 2*pi)
ConditionalValue(2*sqrt(3)*pi/3)
>>> integrate(1/(2 + cos(x)), (x, 0, 2*pi))
2*sqrt(3)*pi/3
>>> antiderivative_integral(1/x**2, x, -1, 1)
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
from __future__ import annotations

import random
from typing import Optional

from sympy.calculus.accumulationbounds import AccumBounds
from sympy.calculus.singularities import singularities
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.function import Function, expand
from sympy.core.numbers import Integer, Rational, nan, oo, pi, zoo
from sympy.core.relational import Relational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.error_functions import erf
from sympy.simplify.powsimp import powsimp
from sympy.functions.elementary.hyperbolic import sinh, cosh
from sympy.functions.elementary.trigonometric import atan, acot, sin, cos
from sympy.polys.polytools import cancel, degree
from sympy.functions.elementary.complexes import Abs, im, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside
from sympy.integrals.integrals import Integral, integrate
from sympy.logic.boolalg import Boolean
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import FiniteSet, Intersection, Interval, Set
from sympy.series.limits import Limit
from sympy.series.series import series
from sympy.core.add import Add

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, as_set, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.limits import limit
from sympy_extras.assumptions.solve import solve
from sympy_extras.settings import settings
from .conditions import ConditionalValue, sample_values

__all__ = ['antiderivative', 'quick_shape', 'by_parts_shape', 'discontinuities', 'antiderivative_integral', 'one_sided_limit',
           'select_branch',
           'principal_value_integral', 'finite_part_integral']


#: the methods of the indefinite dispatcher tried for a definite integral:
#: the exact ones, which fail fast; the heuristics (the Risch–Norman
#: method, the substitutions, SymPy's manual and Meijer routes) spend
#: their whole budget on every piece of every mapped range
_METHODS = ['rational', 'radicals', 'exponential', 'trigonometric', 'risch', 'trager', 'sympy']


#: the heuristic methods, tried late and under a larger budget
_LATE_METHODS = ['heurisch', 'rewriting', 'manual', 'meijer', 'sympy']


def antiderivative(f: Expr, x: Symbol, assumptions: Assumptions = None, late: bool = False) -> Optional[Expr]:
    """An antiderivative of ``f`` by the radical table, by ``integrate``
    for a polynomial in elementary functions, and by the verified methods
    of :mod:`.indefinite` otherwise, under an eighth of the time limit;
    ``None`` when none is found (an unevaluated ``Integral`` in the result
    counts as none). The assumptions on the parameters reach the checks
    (``1/(cosh(n*t)**2 + 1)`` has an antiderivative for ``n > 0``, none
    that checks for a general ``n``). With ``late`` the heuristic methods
    are tried instead, under a quarter of the time limit (the second pass
    of the definite driver, after its slow methods)."""
    from .radicals import quadratic_radical_antiderivative
    found = quadratic_radical_antiderivative(f, x)
    if found is not None:
        # a real form (SymPy writes the arcsine of sqrt(1 - x**2) as a
        # complex logarithm, whose limits at algebraic bounds fail)
        return found
    if _elementary_polynomial(f, x):
        # a polynomial in x and in exponentials and trigonometric functions
        # of linear arguments: integrate does it at once, where the Risch
        # port spends seconds on gcds over the constants of the arguments
        # (sin(pi*t/4 + pi/4)**3 took 7 s, on every quarter period); the
        # arguments expanded, since sin(pi*(t/4 + 1/4)) takes integrate
        # 8 s where sin(pi*t/4 + pi/4) takes a tenth of one
        g = as_expr(f.replace(lambda n: isinstance(n, (sin, cos, exp, sinh, cosh)),
                              lambda n: n.func(expand(as_expr(n.args[0])))))
        found = attempt(lambda: as_expr(integrate(g, x, risch=False)), settings.timeout)
        if found is not None and not found.has(Integral):
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
    # Risch port gives up and integrate answers a Piecewise in tanh
    from .indefinite import verified_antiderivative
    # an eighth of the time limit: the methods may spend it all, the
    # route is tried on every piece of every mapped range, and the other
    # methods of the driver need their share (log(sin(x)/x) over
    # (0, pi/2) lost its budget to the failures on its pieces)
    share = 4 if late else 8
    budget = None if settings.timeout is None else settings.timeout / share
    try:
        verified = attempt(lambda: verified_antiderivative(f, x, assumptions, _LATE_METHODS if late else _METHODS),
                           budget)
    except (AttributeError, ZeroDivisionError, AssertionError):
        # SymPy 1.14: the cache wrapper of meijerint fails on a lazy
        # exception message ('LazyExceptionMessage' has no 'startswith')
        return None
    if verified is None or verified[0].has(Integral):
        return None
    return verified[0]


def _elementary_polynomial(f: Expr, x: Symbol) -> bool:
    """Whether ``f`` is a polynomial in ``x`` and in sines, cosines,
    exponentials and hyperbolic functions of arguments linear in ``x``."""
    replacement: dict[Expr, Expr] = {}
    for node in f.atoms(Function):
        if not node.has(x):
            continue
        if not isinstance(node, (sin, cos, exp, sinh, cosh)) or len(node.args) != 1:
            return False
        argument = as_expr(node.args[0])
        if not argument.is_polynomial(x) or degree(argument, x) != 1:
            return False
        replacement[node] = Dummy()
    substituted = as_expr(f.xreplace(replacement))
    return bool(substituted.is_polynomial(x, *replacement.values()))


def quick_shape(f: Expr, x: Symbol) -> bool:
    """Whether ``f`` is of a shape the antiderivative comes at once for: a
    polynomial in ``x`` and in elementary functions of linear arguments,
    or such a polynomial with exponentials times ``log(x)``."""
    return _elementary_polynomial(f, x) or by_parts_shape(f, x) or _error_function_shape(f, x)


def _error_function_shape(f: Expr, x: Symbol) -> bool:
    errors = [as_expr(factor) for factor in Mul.make_args(f) if isinstance(factor, erf)]
    return len(errors) == 1 and _elementary_polynomial(as_expr(f / errors[0]), x)


def by_parts_shape(f: Expr, x: Symbol) -> bool:
    """Whether ``f`` is a polynomial in ``x`` and exponentials times ``log(x)``."""
    logarithms = [as_expr(factor) for factor in Mul.make_args(f) if isinstance(factor, log)]
    if len(logarithms) != 1 or logarithms[0] != log(x):
        return False
    g = as_expr(f / log(x))
    return not g.has(log) and _elementary_polynomial(g, x) and bool(g.has(exp))


def _error_function_by_parts(f: Expr, x: Symbol) -> Optional[Expr]:
    """``g(x)*erf(k*x)`` for a polynomial ``g`` in ``x`` and in elementary
    functions of linear arguments, by parts: ``G*erf(k*x) -
    2*k/sqrt(pi)*Integral(G*exp(-k**2*x**2))``, the last integrated term
    by term with the exponentials combined (``exp(-w**2 - w)`` completes
    its square at once, where ``integrate`` spent a minute on
    ``cosh(u - w)*exp(-w**2)``): ``sinh(u - w)*erf(w)`` over ``(0, u)``
    (Maxima's ``laplace`` 57). ``None`` for another shape."""
    errors = [as_expr(factor) for factor in Mul.make_args(f) if isinstance(factor, erf)]
    if len(errors) != 1:
        return None
    argument = as_expr(errors[0].args[0])
    k, variable = argument.as_independent(x, as_Add=False)
    if as_expr(variable) != x or as_expr(k).has(x):
        return None
    g = as_expr(f / errors[0])
    if g.has(erf) or not _elementary_polynomial(g, x):
        return None
    plain = Dummy(x.name)
    g = as_expr(g.subs(x, plain))
    budget = None if settings.timeout is None else settings.timeout / 8
    G = attempt(lambda: as_expr(integrate(g, plain, risch=False)), budget)
    if G is None or G.has(Integral):
        return None
    remainder = as_expr(expand(G.rewrite(exp) * exp(-as_expr(k)**2 * plain**2)))
    H: Expr = S.Zero
    for term in Add.make_args(remainder):
        combined = as_expr(powsimp(as_expr(term), combine='exp'))
        piece = attempt(lambda: as_expr(integrate(combined, plain, risch=False)), budget)
        if piece is None or piece.has(Integral):
            return None
        H = H + piece
    return as_expr((G * erf(as_expr(k) * plain) - 2 * as_expr(k) / sqrt(pi) * H).subs(plain, x))


def _logarithm_by_parts(f: Expr, x: Symbol) -> Optional[Expr]:
    """``g(x)*log(x)`` for a polynomial ``g`` in ``x`` and in exponentials
    of linear arguments, by parts: ``G*log(x) - Integral(G/x)`` with ``G``
    the antiderivative of ``g``, the last an exponential integral
    (``u**3*exp(-u)*log(u)`` in two seconds, where ``integrate`` took
    six). ``None`` for another shape."""
    if not by_parts_shape(f, x):
        return None
    g = as_expr(f / log(x))
    # on a plain dummy: integrate is four times slower on a symbol
    # declared positive (the assumptions are consulted at every step)
    plain = Dummy(x.name)
    g = as_expr(g.subs(x, plain))
    budget = None if settings.timeout is None else settings.timeout / 8
    G = attempt(lambda: as_expr(integrate(g, plain, risch=False)), budget)
    if G is None or G.has(Integral):
        return None
    H = attempt(lambda: as_expr(integrate(cancel(G / plain), plain, risch=False)), budget)
    if H is None or H.has(Integral):
        return None
    return as_expr((G * log(plain) - H).subs(plain, x))


def _not_real(p: Expr, assumptions: Assumptions) -> bool:
    """Whether ``p`` is not real once the parameters the assumptions make
    real are taken as such: ``(log(3 - 2*sqrt(2)) + I*pi)/(2*n)`` for
    ``n > 0`` (a singularity of an antiderivative in ``exp(2*n*t)`` which
    lies off the real line, where SymPy's ``singularities`` leaves it
    intersected with the range)."""
    replacement: dict[Expr, Expr] = {}
    for q in p.free_symbols:
        if not isinstance(q, Symbol):
            continue
        if ask(as_boolean(q > 0), assumptions) is True:
            replacement[q] = Dummy(q.name, positive=True)
        elif ask(as_boolean(q < 0), assumptions) is True:
            replacement[q] = Dummy(q.name, negative=True)
        elif q.is_extended_real or ask(element(q, S.Reals), assumptions):
            replacement[q] = Dummy(q.name, real=True)
    if not replacement:
        return False
    substituted = as_expr(p.xreplace(replacement))
    if substituted.is_extended_real is False:
        return True
    imaginary = as_expr(im(substituted))
    return not imaginary.has(im) and imaginary.is_zero is False


def _points_in(points: list[Expr], x: Symbol, a: Expr, b: Expr,
               assumptions: Assumptions) -> Optional[list[Expr]]:
    """The points strictly inside ``(a, b)``, sorted; ``None`` when a
    position is undecided."""
    inside: list[Expr] = []
    for p in points:
        if p.is_extended_real is False or _not_real(p, assumptions):
            continue
        below = True if a == -oo else ask(as_boolean(p > a), assumptions)
        above = True if b == oo else ask(as_boolean(p < b), assumptions)
        if below is None or above is None:
            return None
        if below and above and p not in inside:
            inside.append(p)
    for i in range(len(inside)):
        for j in range(i + 1, len(inside)):
            if ask(as_boolean(inside[i] < inside[j]), assumptions) is None:
                return None
    return sorted(inside, key=lambda p: sum(1 for q in inside if ask(as_boolean(q < p), assumptions)))


def discontinuities(F: Expr, x: Symbol, a: Expr, b: Expr,
                    assumptions: Assumptions = None) -> Optional[list[Expr]]:
    """The points of ``(a, b)`` where the antiderivative ``F`` may be
    discontinuous: its singularities (poles, zeros of the arguments of
    logarithms) and the poles of the arguments of ``atan`` and ``acot``
    (where ``atan`` jumps by ``pi``), together with the kinks of
    ``Abs``, ``sign``, ``Heaviside`` and ``Piecewise``; ``None`` when
    they cannot be located."""
    candidates: list[Expr] = []
    found = attempt(lambda: singularities(F, x, Interval.open(a, b)), settings.timeout)
    if found is None:
        return None
    found_set = as_set(found)
    if isinstance(found_set, FiniteSet):
        candidates.extend(as_expr(p) for p in found_set)
    elif found_set.has(ConditionSet, ImageSet):
        return None
    elif found_set is not S.EmptySet:
        # a union of points intersected with the range, which SymPy
        # cannot place for symbolic parameters: placed below
        for finite in found_set.atoms(FiniteSet):
            candidates.extend(as_expr(p) for p in finite)
    arguments: list[Expr] = []
    for node in F.atoms(atan, acot):
        if node.has(x):
            u = as_expr(node.args[0])
            _, denominator = u.as_numer_denom()
            if as_expr(denominator).has(x):
                arguments.append(as_expr(denominator))
    for node in F.atoms(log):
        if node.has(x):
            arguments.append(as_expr(node.args[0]))
    for node in F.atoms(Abs, sign, Heaviside):
        if node.has(x):
            arguments.append(as_expr(node.args[0]))
    for node in F.atoms(Piecewise):
        for pair in node.args:
            for relation in as_boolean(pair.args[1]).atoms(Boolean):
                lhs = as_expr(relation.args[0]) if len(relation.args) == 2 else None
                rhs = as_expr(relation.args[1]) if len(relation.args) == 2 else None
                if lhs is not None and rhs is not None and (lhs - rhs).has(x):
                    arguments.append(lhs - rhs)
    bounded = _bounded(assumptions, x, a, b)
    for u in arguments:
        if ask(as_boolean(u > 0), bounded) is True or ask(as_boolean(u < 0), bounded) is True:
            # no zero on the range (exp(2*n*t) - 2*sqrt(2) + 3 for n > 0,
            # where the solver answers a ConditionSet)
            continue
        zeros = attempt(lambda: solve(u, x, bounded, domain=S.Reals), settings.timeout)
        if zeros is None:
            return None
        if zeros is S.EmptySet:
            continue
        zero_set = as_set(zeros)
        if not isinstance(zero_set, FiniteSet):
            return None
        candidates.extend(as_expr(p) for p in zero_set)
    return _points_in(candidates, x, a, b, assumptions)


def _bounded(assumptions: Assumptions, x: Symbol, a: Expr, b: Expr) -> list[Boolean]:
    facts: list[Boolean] = []
    if a != -oo:
        facts.append(as_boolean(x > a))
    if b != oo:
        facts.append(as_boolean(x < b))
    if isinstance(assumptions, (Boolean, bool)):
        facts.append(as_boolean(assumptions))
    elif assumptions is not None:
        facts.extend(as_boolean(s) for s in assumptions)
    return facts


def select_branch(F: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions = None) -> Optional[Expr]:
    """The branch of a ``Piecewise`` antiderivative which holds on
    ``(a, b)``, decided at a sample point of the range under the
    assumptions (the conditions of SymPy's antiderivatives are on the
    variable, ``x > 1``); ``None`` when a condition cannot be decided or
    the branches change inside the range (the bug: ``limit`` of a
    ``Piecewise`` with an infinite value on the branch of the range gave
    0, and the area of a disc came out as 0)."""
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
            value, condition = as_expr(pair.args[0]), as_boolean(pair.args[1])
            if condition is S.true:
                chosen = value
                break
            # the same branch must hold on the whole range: no condition
            # met on the way may change inside it
            for atom in condition.atoms(Relational):
                zeros = attempt(lambda: solve(as_expr(atom.lhs) - as_expr(atom.rhs), x,
                                              _bounded(assumptions, x, a, b), domain=S.Reals), settings.timeout)
                if zeros is None or zeros is not S.EmptySet:
                    return None
            holds = ask(as_boolean(condition.subs(x, sample)), assumptions)
            if holds is None:
                return None
            if holds:
                chosen = value
                break
        if chosen is None:
            return None
        result = as_expr(result.xreplace({node: chosen}))
    return result


def one_sided_limit(F: Expr, x: Symbol, point: Expr, direction: str,
                    assumptions: Assumptions = None) -> Optional[Expr]:
    """``lim F(x)`` as ``x`` tends to ``point`` from the side ``'+'`` or
    ``'-'`` (``F(point)`` when ``F`` is continuous there); ``oo`` or
    ``-oo`` for a signed infinity, and ``None`` when the limit does not
    exist (an oscillation, ``AccumBounds``), is complex infinite, or
    cannot be found."""
    if point in (oo, -oo):
        value = attempt(lambda: limit(F, x, point, assumptions=assumptions), settings.timeout)
    else:
        value = attempt(lambda: limit(F, x, point, direction, assumptions=assumptions), settings.timeout)
        if (value is None or value.has(Limit)) and not point.is_number:
            # a symbolic point, where limit gives up: the value of F there
            # when it is finite (the bug: the arcsine of a real antiderivative
            # had no limit at the algebraic bound -sqrt(1 - x**2) of a cell)
            value = as_expr(F.subs(x, point))
    if value is None:
        return None
    if value in (oo, -oo):
        return value
    if value.has(oo, -oo, zoo, nan, Limit, Piecewise, AccumBounds):
        return None
    if value.free_symbols - F.free_symbols - point.free_symbols:
        # a dummy of an unevaluated inner limit leaked (the symbols of a
        # symbolic point are its own: the bug refused every limit at the
        # algebraic bound -sqrt(1 - x**2) of a cell)
        return None
    return value


def antiderivative_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                            assumptions: Assumptions = None, late: bool = False) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` from an antiderivative ``F``, cut at the
    discontinuities of ``F`` and the singularities of ``f`` inside the
    range and evaluated by one-sided limits; ``oo`` or ``-oo`` when
    the limits are infinite of one sign (the integral diverges to it, as
    ``1/x**2`` over ``(-1, 1)``); ``None`` when there is no
    antiderivative, a point cannot be placed, a limit is unknown, or the
    infinities cancel (``1/x`` over ``(-1, 1)``, where the principal
    value is another question).

    Examples
    ========

    >>> from sympy import symbols, pi, cos, tan, oo
    >>> from sympy_extras.integrals.antiderivative import antiderivative_integral
    >>> x = symbols('x')
    >>> antiderivative_integral(1/(5 - 4*cos(x)), x, 0, 2*pi)
    ConditionalValue(2*pi/3)
    >>> antiderivative_integral(1/(1 + x**2), x, -oo, oo)
    ConditionalValue(pi)
    >>> antiderivative_integral(1/x, x, 0, 1)
    ConditionalValue(oo)
    """
    a, b = as_expr(a), as_expr(b)
    F = antiderivative(f, x, assumptions, late)
    if F is None:
        return None
    if F.has(Piecewise):
        F = select_branch(F, x, a, b, assumptions)
        if F is None:
            return None
    F = _real_logarithms(F, x)
    points = discontinuities(F, x, a, b, assumptions)
    if points is None:
        return None
    singular = attempt(lambda: singularities(f, x, Interval.open(a, b)), settings.timeout)
    if singular is None and not (a.is_number and b.is_number):
        # SymPy cannot form an interval with symbolic ends (the algebraic
        # bounds -sqrt(1 - x**2) < y < x of a cell): the singularities on
        # the line, placed against the bounds by _points_in below
        singular = attempt(lambda: singularities(f, x, S.Reals), settings.timeout)
    if singular is None:
        return None
    singular_points = _singular_points(as_set(singular))
    if singular_points is None:
        return None
    more = _points_in(singular_points + points, x, a, b, assumptions)
    if more is None:
        return None
    bounds = [a] + more + [b]
    total: Expr = S.Zero
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        right = one_sided_limit(F, x, upper, '-', assumptions)
        left = one_sided_limit(F, x, lower, '+', assumptions)
        if right is None or left is None:
            return None
        # an infinite limit is checked against the sign of the integrand
        # near the point (SymPy's limit of x**2*Shi(x)/2 - x*cosh(x)/2 +
        # sinh(x)/2 at oo is -oo; the function grows like x*exp(x)/4)
        if right in (oo, -oo) and not _integrand_has_sign(f, x, upper, '-', right, assumptions):
            return None
        if left in (oo, -oo) and not _integrand_has_sign(f, x, lower, '+', -left, assumptions):
            return None
        total = total + (right - left)
    if total.has(nan, zoo):
        # oo - oo: infinities of both signs, nothing is claimed
        return None
    return ConditionalValue(_absorbed(as_expr(total)))


def _real_logarithms(F: Expr, x: Symbol) -> Expr:
    """``log(u)`` written ``log(Abs(u))`` for the real arguments ``u``
    depending on ``x``: the same antiderivative where ``u`` keeps its
    sign, real where ``u`` is negative (``log(sin(x)/tan(1) - cos(x))``
    on ``(0, 1)`` is ``log(-u) + I*pi``, whose infinite limit at 1 the
    complex values make ``zoo``)."""
    real = Dummy(x.name, real=True)
    return as_expr(F.replace(
        lambda node: isinstance(node, log) and node.has(x) and not isinstance(node.args[0], Abs)
        and as_expr(node.args[0]).subs(x, real).is_extended_real is True,
        lambda node: log(Abs(as_expr(node.args[0])))))


def _singular_points(singular_set: Set) -> Optional[list[Expr]]:
    """The points of a set of singularities: those of a finite set, none
    of the empty set, and those of the finite part of ``Intersection({0},
    Interval.open(sqrt(x), oo))`` (SymPy's answer for a parameter ``x`` not
    declared positive), which the placement against the bounds decides
    under the assumptions; ``None`` for another kind of set."""
    if isinstance(singular_set, Intersection) and len(singular_set.args) == 2:
        finite = [arg for arg in singular_set.args if isinstance(arg, FiniteSet)]
        intervals = [arg for arg in singular_set.args if isinstance(arg, Interval)]
        if len(finite) == 1 and len(intervals) == 1:
            singular_set = finite[0]
    if isinstance(singular_set, FiniteSet):
        return [as_expr(p) for p in singular_set]
    if singular_set is S.EmptySet:
        return []
    return None


def _absorbed(total: Expr) -> Expr:
    """``total`` with the finite numbers of a sum absorbed into its
    infinity: SymPy keeps ``-Si(1)/2 + oo`` as a sum (the finiteness of
    ``Si(1)`` is not known to it), which is ``oo`` (``sin(x)/x**3`` over
    ``(0, 1)``)."""
    if total in (oo, -oo) or not total.has(oo, -oo):
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
    parameters = sorted_symbols((free_symbols(f) | free_symbols(point)) - {x})
    values = sample_values(parameters, assumptions, random.Random(str((f, point)))) if parameters else {}
    if values is None:
        return True
    g, p = as_expr(f.xreplace(values)), as_expr(point.xreplace(values))
    if p is oo:
        samples: list[Expr] = [Integer(10), Integer(100), Integer(1000)]
    elif p is -oo:
        samples = [Integer(-10), Integer(-100), Integer(-1000)]
    else:
        step = 1 if side == '+' else -1
        samples = [p + step * Rational(1, 10**k) for k in (2, 4, 6)]
    sign = 1 if infinity is oo else -1
    for sample in samples:
        try:
            value = as_expr(g.subs(x, sample).evalf(15))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return True
        if not value.is_number or value.is_extended_real is not True:
            return True
        if value * sign < 0:
            return False
    return True


def principal_value_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                             assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """Cauchy's principal value of ``Integral(f, (x, a, b))`` from an
    antiderivative ``F``: with `c_1, \\ldots, c_n` the singularities of
    ``f`` and the discontinuities of ``F`` inside the range,

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

    >>> from sympy import symbols, log
    >>> from sympy_extras.integrals.antiderivative import principal_value_integral
    >>> x = symbols('x')
    >>> principal_value_integral(1/x, x, -1, 2)
    ConditionalValue(log(2))
    >>> principal_value_integral(1/x**2, x, -1, 1) is None
    True
    """
    a, b = as_expr(a), as_expr(b)
    F = antiderivative(f, x, assumptions)
    if F is None:
        return None
    if F.has(Piecewise):
        F = select_branch(F, x, a, b, assumptions)
        if F is None:
            return None
    points = discontinuities(F, x, a, b, assumptions)
    if points is None:
        return None
    singular = attempt(lambda: singularities(f, x, Interval.open(a, b)), settings.timeout)
    if singular is None and not (a.is_number and b.is_number):
        # SymPy cannot form an interval with symbolic ends (the algebraic
        # bounds -sqrt(1 - x**2) < y < x of a cell): the singularities on
        # the line, placed against the bounds by _points_in below
        singular = attempt(lambda: singularities(f, x, S.Reals), settings.timeout)
    if singular is None:
        return None
    singular_points = _singular_points(as_set(singular))
    if singular_points is None:
        return None
    inside = _points_in(singular_points + points, x, a, b, assumptions)
    if inside is None:
        return None
    right = one_sided_limit(F, x, b, '-', assumptions)
    left = one_sided_limit(F, x, a, '+', assumptions)
    if right is None or left is None or right in (oo, -oo) or left in (oo, -oo):
        # the principal value and the finite part take finite limits at
        # the endpoints only
        return None
    total: Expr = right - left
    epsilon = Dummy('epsilon', positive=True)
    for c in inside:
        jump = attempt(lambda: limit(F.subs(x, c + epsilon) - F.subs(x, c - epsilon), epsilon, 0,
                                     assumptions=assumptions), settings.timeout)
        if jump is None or jump.has(oo, -oo, zoo, nan, Limit) or jump.free_symbols - F.free_symbols:
            return None
        total = total - jump
    return ConditionalValue(as_expr(total))


def _finite_part(jump: Expr, epsilon: Symbol) -> Optional[Expr]:
    """The constant term of the expansion of ``jump`` at ``epsilon = 0``,
    the terms in negative powers of ``epsilon`` and in ``log(epsilon)``
    dropped (Hadamard's finite part)."""
    expansion = attempt(lambda: as_expr(series(jump, epsilon, 0, 1).removeO()), settings.timeout)
    if expansion is None:
        return None
    constant: Expr = S.Zero
    for term in Add.make_args(as_expr(expansion.expand())):
        term_ = as_expr(term)
        if not term_.has(epsilon):
            constant = constant + term_
    if constant.has(nan, zoo, oo, -oo):
        return None
    return constant


def finite_part_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                         assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """Hadamard's finite part of ``Integral(f, (x, a, b))``: as in
    :func:`principal_value_integral`, but the divergent terms of the
    symmetric excision at each singularity (``1/epsilon``,
    ``log(epsilon)``) are dropped instead of required to cancel, which
    regularises a double pole (``Integral(1/x**2, (x, -1, 1))`` has the
    finite part ``-2``) [Hadamard]_.

    Examples
    ========

    >>> from sympy import symbols, log
    >>> from sympy_extras.integrals.antiderivative import finite_part_integral
    >>> x = symbols('x')
    >>> finite_part_integral(1/x**2, x, -1, 1)
    ConditionalValue(-2)
    >>> finite_part_integral(1/x, x, -1, 2)
    ConditionalValue(log(2))

    References
    ==========

    .. [Hadamard] J. Hadamard, *Lectures on Cauchy's problem in linear
       partial differential equations*, Yale University Press, 1923,
       chapter on the "partie finie"; the modern account in Estrada and
       Kanwal, *Singular Integral Equations*, Birkhäuser, 2000, chapter 2.
    """
    a, b = as_expr(a), as_expr(b)
    F = antiderivative(f, x, assumptions)
    if F is None:
        return None
    if F.has(Piecewise):
        F = select_branch(F, x, a, b, assumptions)
        if F is None:
            return None
    points = discontinuities(F, x, a, b, assumptions)
    if points is None:
        return None
    singular = attempt(lambda: singularities(f, x, Interval.open(a, b)), settings.timeout)
    if singular is None and not (a.is_number and b.is_number):
        # SymPy cannot form an interval with symbolic ends (the algebraic
        # bounds -sqrt(1 - x**2) < y < x of a cell): the singularities on
        # the line, placed against the bounds by _points_in below
        singular = attempt(lambda: singularities(f, x, S.Reals), settings.timeout)
    if singular is None:
        return None
    singular_points = _singular_points(as_set(singular))
    if singular_points is None:
        return None
    inside = _points_in(singular_points + points, x, a, b, assumptions)
    if inside is None:
        return None
    right = one_sided_limit(F, x, b, '-', assumptions)
    left = one_sided_limit(F, x, a, '+', assumptions)
    if right is None or left is None or right in (oo, -oo) or left in (oo, -oo):
        # the principal value and the finite part take finite limits at
        # the endpoints only
        return None
    total: Expr = right - left
    epsilon = Dummy('epsilon', positive=True)
    for c in inside:
        jump = _finite_part(as_expr(F.subs(x, c + epsilon) - F.subs(x, c - epsilon)), epsilon)
        if jump is None or jump.free_symbols - F.free_symbols:
            return None
        total = total - jump
    return ConditionalValue(as_expr(total))
