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
>>> antiderivative_integral(1/x**2, x, -1, 1) is None
True

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

from typing import Optional

from sympy.calculus.singularities import singularities
from sympy.core.expr import Expr
from sympy.core.numbers import Rational, nan, oo, zoo
from sympy.core.power import Pow
from sympy.core.relational import Relational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan, acot
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside
from sympy.integrals.integrals import Integral, integrate
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import FiniteSet, Interval
from sympy.series.limits import Limit
from sympy.series.series import series
from sympy.core.add import Add

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, as_set
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.limits import limit
from sympy_extras.assumptions.solve import solve
from sympy_extras.settings import settings
from .conditions import ConditionalValue

__all__ = ['antiderivative', 'discontinuities', 'antiderivative_integral', 'one_sided_limit', 'select_branch',
           'principal_value_integral', 'finite_part_integral']


def antiderivative(f: Expr, x: Symbol) -> Optional[Expr]:
    """An antiderivative of ``f`` by the Risch port when it applies and by
    SymPy's ``integrate`` otherwise, under the time limit; ``None`` when
    none is found (an unevaluated ``Integral`` in the result counts as
    none)."""
    from .radicals import quadratic_radical_antiderivative
    found = quadratic_radical_antiderivative(f, x)
    if found is not None:
        # a real form (SymPy writes the arcsine of sqrt(1 - x**2) as a
        # complex logarithm, whose limits at algebraic bounds fail)
        return found
    found = _risch(f, x)
    if found is not None and not found.has(Integral):
        return found
    found = _trager(f, x)
    if found is not None:
        return found
    # a quarter of the time limit: the heuristics of integrate may spend
    # it all, and the other methods of the driver still need their share
    budget = None if settings.timeout is None else settings.timeout / 4
    try:
        value = attempt(lambda: as_expr(integrate(f, x, risch=False)), budget)
    except (AttributeError, ZeroDivisionError, AssertionError):
        # SymPy 1.14: the cache wrapper of meijerint fails on a lazy
        # exception message ('LazyExceptionMessage' has no 'startswith')
        return None
    if value is None or value.has(Integral):
        return None
    return value


def _risch(f: Expr, x: Symbol) -> Optional[Expr]:
    """The antiderivative by the Risch port, ``None`` when it does not
    apply (or the port is not available)."""
    try:
        from .risch import risch_antiderivative
    except ImportError:
        return None
    budget = None if settings.timeout is None else settings.timeout / 4
    return attempt(lambda: risch_antiderivative(f, x), budget)


def _trager(f: Expr, x: Symbol) -> Optional[Expr]:
    """The antiderivative by Trager's algorithm (:mod:`.trager`) for an
    integrand rational in ``x`` and one square root of a polynomial,
    ``None`` otherwise."""
    if not any(isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 2 and node.has(x)
               for node in f.atoms(Pow)):
        return None
    from .trager import trager_antiderivative
    budget = None if settings.timeout is None else settings.timeout / 4
    return attempt(lambda: trager_antiderivative(f, x), budget)


def _points_in(points: list[Expr], x: Symbol, a: Expr, b: Expr,
               assumptions: Assumptions) -> Optional[list[Expr]]:
    """The points strictly inside ``(a, b)``, sorted; ``None`` when a
    position is undecided."""
    inside: list[Expr] = []
    for p in points:
        if p.is_extended_real is False:
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
    elif found_set is not S.EmptySet:
        return None
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
    for u in arguments:
        zeros = attempt(lambda: solve(u, x, _bounded(assumptions, x, a, b), domain=S.Reals), settings.timeout)
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
    ``'-'`` (``F(point)`` when ``F`` is continuous there), ``None`` when
    the limit is infinite or cannot be found."""
    if point in (oo, -oo):
        value = attempt(lambda: limit(F, x, point, assumptions=assumptions), settings.timeout)
    else:
        value = attempt(lambda: limit(F, x, point, direction, assumptions=assumptions), settings.timeout)
        if (value is None or value.has(Limit)) and not point.is_number:
            # a symbolic point, where limit gives up: the value of F there
            # when it is finite (the bug: the arcsine of a real antiderivative
            # had no limit at the algebraic bound -sqrt(1 - x**2) of a cell)
            value = as_expr(F.subs(x, point))
    if value is None or value.has(oo, -oo, zoo, nan, Limit, Piecewise):
        return None
    if value.free_symbols - F.free_symbols - point.free_symbols:
        # a dummy of an unevaluated inner limit leaked (the symbols of a
        # symbolic point are its own: the bug refused every limit at the
        # algebraic bound -sqrt(1 - x**2) of a cell)
        return None
    return value


def antiderivative_integral(f: Expr, x: Symbol, a: ExprLike, b: ExprLike,
                            assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` from an antiderivative ``F``, cut at the
    discontinuities of ``F`` and the singularities of ``f`` inside the
    range and evaluated by one-sided limits; ``None`` when there is no
    antiderivative, a point cannot be placed, or a limit is infinite
    (the integral diverges) or unknown.

    Examples
    ========

    >>> from sympy import symbols, pi, cos, tan, oo
    >>> from sympy_extras.integrals.antiderivative import antiderivative_integral
    >>> x = symbols('x')
    >>> antiderivative_integral(1/(5 - 4*cos(x)), x, 0, 2*pi)
    ConditionalValue(2*pi/3)
    >>> antiderivative_integral(1/(1 + x**2), x, -oo, oo)
    ConditionalValue(pi)
    """
    a, b = as_expr(a), as_expr(b)
    F = antiderivative(f, x)
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
    singular_set = as_set(singular)
    if isinstance(singular_set, FiniteSet):
        singular_points = [as_expr(p) for p in singular_set]
    elif singular_set is S.EmptySet:
        singular_points = []
    else:
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
        total = total + (right - left)
    return ConditionalValue(as_expr(total))


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
    F = antiderivative(f, x)
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
    singular_set = as_set(singular)
    if isinstance(singular_set, FiniteSet):
        singular_points = [as_expr(p) for p in singular_set]
    elif singular_set is S.EmptySet:
        singular_points = []
    else:
        return None
    inside = _points_in(singular_points + points, x, a, b, assumptions)
    if inside is None:
        return None
    right = one_sided_limit(F, x, b, '-', assumptions)
    left = one_sided_limit(F, x, a, '+', assumptions)
    if right is None or left is None:
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
    F = antiderivative(f, x)
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
    singular_set = as_set(singular)
    if isinstance(singular_set, FiniteSet):
        singular_points = [as_expr(p) for p in singular_set]
    elif singular_set is S.EmptySet:
        singular_points = []
    else:
        return None
    inside = _points_in(singular_points + points, x, a, b, assumptions)
    if inside is None:
        return None
    right = one_sided_limit(F, x, b, '-', assumptions)
    left = one_sided_limit(F, x, a, '+', assumptions)
    if right is None or left is None:
        return None
    total: Expr = right - left
    epsilon = Dummy('epsilon', positive=True)
    for c in inside:
        jump = _finite_part(as_expr(F.subs(x, c + epsilon) - F.subs(x, c - epsilon)), epsilon)
        if jump is None or jump.free_symbols - F.free_symbols:
            return None
        total = total - jump
    return ConditionalValue(as_expr(total))
