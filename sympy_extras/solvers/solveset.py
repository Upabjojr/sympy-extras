"""SymPy's ``solveset`` of one relation in one unknown over the real or
the complex numbers, on the objects of the core.

:func:`solveset` returns the solution set of a relation for ``x`` in a
domain, a :class:`~sympy_extras.sets.conditionset.ConditionSet` of the
relation when it is not solved (SymPy's answer then). The cases are those
of SymPy's ``_solveset`` which the callers of this package meet:

* over the reals, a relation rational in ``x`` with numbers as
  coefficients: the roots of the numerator (all of them, in radicals),
  the real ones told apart exactly or by reliable values (a real
  polynomial has its non-real roots in conjugate pairs, so that an
  isolated root with a negligible imaginary part is real), and the signs
  between the real roots for an inequality; with parameters, the roots
  of an equation (those not real for every value of the parameters
  intersected with the reals), a linear inequality whose coefficient has
  a known sign and the even binomials ``x**(2k) op c``;
* over the reals, a transcendental relation by
  :func:`~sympy_extras.solvers.transcendental.solve_transcendental` (its
  polynomial relations with constant coefficients solved by the case
  above), and an equation without trigonometric or hyperbolic functions
  of ``x`` by :func:`~sympy_extras.solvers.solvers.solve_equation`, its
  real candidates checked;
* over the complex numbers, a rational equation (the roots of the
  numerator which are not roots of the denominator), an equation
  ``g(a*x + b) = c`` for an exponential, a trigonometric or a hyperbolic
  function ``g`` (the inverse images, SymPy's ``_invert_complex``, as
  images of the integers), and an algebraic equation by
  :func:`~sympy_extras.solvers.solvers.solve_equation` with the
  candidates which provably fail dropped.

The module imports no SymPy. It is a port of parts of SymPy's
``sympy/solvers/solveset.py``.

Examples
========

>>> from sympy_extras.core import Eq, Symbol, sqrt, I
>>> from sympy_extras.functions import cos, sinh
>>> from sympy_extras.sets import Reals, Complexes
>>> from sympy_extras.solvers.solveset import solveset
>>> x = Symbol('x')
>>> solveset(Eq(x**2*(sqrt(2) - 1)**2 - 1, 0), x, Reals)
{-sqrt(-1/(-3 + 2*sqrt(2))), sqrt(-1/(-3 + 2*sqrt(2)))}
>>> solveset(x**2 - 2 > sqrt(2), x, Reals)
Union(Interval.open(-oo, -sqrt(sqrt(2) + 2)), Interval.open(sqrt(sqrt(2) + 2), oo))
>>> solveset(Eq(sinh(x), I), x, Complexes)
ImageSet(Lambda(_n, 2*_n*I*pi + I*pi/2), Integers)
>>> solveset(Eq(x, cos(x)), x, Reals)
ConditionSet(x, Eq(x, cos(x)), Reals)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/solveset.py (_solveset, _invert_complex,
# _solve_as_rational, _solve_radical, _transolve) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; one relation; the real roots told apart by
# reliable values; the transcendental relations by the native
# solve_transcendental; a ConditionSet instead of NotImplementedError.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import Application, diff
from sympy_extras.core.logic import And, Boolean, BooleanFalse, BooleanTrue, Or
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import I, Integer, NegativeInfinity, Rational, oo, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Relational
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, log
from sympy_extras.sets.conditionset import condition_set
from sympy_extras.sets.fancysets import ComplexesClass, Integers, image_set
from sympy_extras.sets.sets import (EmptySet, Reals, RealsClass, Set, finite_set, intersection,
    interval, union)
from sympy_extras.settings import settings

__all__ = ['solveset', 'real_polynomial_solutions']

_ZERO = Integer(0)
_ONE = Integer(1)


def _expr(b: Basic) -> Expr:
    return as_expr(b)


def _fraction(f: Expr) -> tuple[Expr, Expr]:
    from sympy_extras.polys.rationaltools import together
    from sympy_extras.simplify.radsimp import fraction
    return fraction(together(f))


def _flipped(op: str) -> str:
    return {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '==', '!=': '!='}[op]


def _expanded_numbers(found: list[Expr]) -> list[Expr]:
    """The roots which are numbers written ``a + b*I`` (SymPy's
    ``_solve_as_poly`` applies ``expand_complex`` to them): the formulas
    of Ferrari put a square root on its branch cut, where a numerical
    evaluation picks either branch."""
    if any(r.free_symbols for r in found):
        return [_distributed(r) if r.as_numer_denom()[1] != _ONE else r for r in found]
    from sympy_extras.core.function import expand_complex
    return [expand_complex(r) for r in found]


def _distributed(r: Expr) -> Expr:
    """A root with parameters with its quotient distributed when that is
    shorter, as SymPy's ``roots_quadratic`` writes it:
    ``-k - k/sqrt(c)`` for ``(-k*sqrt(c) - k)/sqrt(c)`` (the assumptions
    decide the short form much faster at the points of a solution set)."""
    from sympy_extras.core.function import count_ops, expand_mul
    candidate = expand_mul(r)
    shorter = count_ops(candidate)
    longer = count_ops(r)
    if isinstance(shorter, Integer) and isinstance(longer, Integer) and shorter.p < longer.p:
        return candidate
    return r


# ---------------------------------------------------------------------------
# the real roots

def _numeric(e: Expr) -> Optional[complex]:
    if not settings.numerical_checks:
        return None
    from sympy_extras._numeric import reliable_complex
    return attempt(lambda: reliable_complex(e, settings.precision), settings.timeout, native=True)


def _real_roots(found: list[Expr], real_coefficients: bool) -> Optional[tuple[list[Expr], list[Expr]]]:
    """The real roots among all the distinct roots of a polynomial, and
    those undecided; ``None`` when a value is not reliable. A root is real
    by the assumptions, or when its imaginary part is negligible and no
    other root lies near it (the conjugate of a non-real root of a real
    polynomial is another root)."""
    real: list[Expr] = []
    undecided: list[Expr] = []
    values: list[Optional[complex]] = [None if r.is_real is not None else _numeric(r) for r in found]
    tolerance = 10.0**(-(settings.precision//2))
    for r, value in zip(found, values):
        if r.is_real is True:
            real.append(r)
            continue
        if r.is_real is False:
            continue
        if value is None:
            undecided.append(r)
            continue
        if abs(value.imag) > tolerance*max(1.0, abs(value)):
            continue
        isolated = all(other is None or other is value or abs(other - value) > 4*tolerance*max(1.0, abs(value))
                       for other in values)
        if real_coefficients and isolated:
            real.append(r)
        else:
            undecided.append(r)
    return real, undecided


def _sorted_real(points: list[Expr]) -> Optional[list[Expr]]:
    """The real numbers in increasing order (by certified comparisons)."""
    from sympy_extras.solvers.isolation import compare_points
    result: list[Expr] = []
    for p in points:
        position = 0
        while position < len(result):
            s = compare_points(p, result[position])
            if s is None:
                return None
            if s < 0:
                break
            position += 1
        result.insert(position, p)
    return result


def _sign_at(f: Expr, x: Symbol, point: Expr) -> Optional[int]:
    from sympy_extras.assumptions.analysis import certified_sign
    value = f.subs(x, point)
    if not isinstance(value, Expr):
        return None
    s = certified_sign(value)
    return None if s is None else int(s)


def _sample(p: Expr, q: Expr) -> Optional[Expr]:
    """A rational point between two real numbers (or beyond one of them
    when the other is infinite)."""
    from sympy_extras.solvers.isolation import point_between
    return point_between(p, q)


def real_polynomial_solutions(f: Expr, x: Symbol, op: str) -> Optional[Set]:
    """The real solutions of ``f op 0`` for ``f`` rational in ``x`` with
    numbers as coefficients (all the roots of the numerator found, the
    real ones decided); ``None`` otherwise.

    >>> from sympy_extras.core import Symbol, sqrt, pi
    >>> from sympy_extras.solvers.solveset import real_polynomial_solutions
    >>> x = Symbol('x')
    >>> real_polynomial_solutions(x**3 - pi, x, '==')
    {pi**(1/3)}
    >>> real_polynomial_solutions(x**2 - sqrt(2), x, '<')
    Interval.open(-2**(1/4), 2**(1/4))
    """
    from .solvers import _coefficients, _polynomial_roots
    numerator, denominator = _fraction(f)
    if numerator.free_symbols - {x} or denominator.free_symbols - {x}:
        return None
    coefficients = _coefficients(numerator, x)
    den_coefficients = _coefficients(denominator, x)
    if coefficients is None or den_coefficients is None:
        return None
    if not coefficients:
        return Reals if op in ('==', '<=', '>=') else EmptySet
    degree = max(coefficients)
    real_coefficients = all(c.is_real is True for c in list(coefficients.values()) + list(den_coefficients.values()))
    roots: list[Expr] = []
    if degree > 0:
        found = _polynomial_roots(numerator, x, degree)
        if found is None:
            return None
        decided = _real_roots(_expanded_numbers(found), real_coefficients)
        if decided is None:
            return None
        roots, undecided = decided
        if undecided:
            return None
    poles: list[Expr] = []
    if max(den_coefficients) > 0:
        found_poles = _polynomial_roots(denominator, x, max(den_coefficients))
        if found_poles is None:
            return None
        decided_poles = _real_roots(found_poles, real_coefficients)
        if decided_poles is None or decided_poles[1]:
            return None
        poles = decided_poles[0]
    from .solvers import zero_truth
    roots = [r for r in roots if not any(zero_truth(r - p) is True for p in poles)]
    if op == '==':
        return finite_set(*roots) if roots else EmptySet
    if not real_coefficients:
        return None
    ordered_points = _sorted_real(roots + poles)
    if ordered_points is None:
        return None
    if op == '!=':
        return _complement_of_points(ordered_points, roots)
    wanted = 1 if op in ('>', '>=') else -1
    strict = op in ('>', '<')
    bounds: list[Expr] = [NegativeInfinity()] + ordered_points + [oo]
    parts: list[Set] = []
    for p, q in zip(bounds, bounds[1:]):
        point = _sample(p, q)
        if point is None:
            return None
        s = _sign_at(f, x, point)
        if s is None:
            return None
        if s == wanted:
            parts.append(interval(p, q, True, True))
    if not strict:
        parts.extend(finite_set(r) for r in roots)
    return union(*parts) if parts else EmptySet


def _complement_of_points(points: list[Expr], removed: list[Expr]) -> Set:
    """The reals without the points ``removed`` (the others, the poles,
    are not solutions either)."""
    bounds: list[Expr] = [NegativeInfinity()] + points + [oo]
    return union(*[interval(p, q, True, True) for p, q in zip(bounds, bounds[1:])])


# ---------------------------------------------------------------------------
# polynomials with parameters over the reals

def _parametric_real(f: Expr, x: Symbol, op: str) -> Optional[Set]:
    from .solvers import _coefficients, _polynomial_roots
    numerator, denominator = _fraction(f)
    coefficients = _coefficients(numerator, x)
    if coefficients is None or any(c.has(x) for c in coefficients.values()):
        return None
    if denominator.has(x):
        return None
    if not coefficients:
        return Reals if op in ('==', '<=', '>=') else EmptySet
    degree = max(coefficients)
    if op == '==':
        if degree == 0:
            return EmptySet
        found = _polynomial_roots(numerator, x, degree)
        if found is None:
            return None
        kept = [r for r in found if r.is_real is True]
        undecided = [r for r in found if r.is_real is None]
        result = finite_set(*kept) if kept else EmptySet
        if undecided:
            result = union(result, intersection(finite_set(*undecided), Reals))
        return result
    if any(c.is_real is not True for c in coefficients.values()):
        return None
    lead = coefficients[degree]
    if lead.is_positive is not True and lead.is_negative is not True:
        return None
    if lead.is_negative is True:
        op = _flipped(op)
    if degree == 1:
        root = -coefficients.get(0, _ZERO)/lead
        if op in ('>', '>='):
            return interval(root, oo, op == '>', True)
        if op in ('<', '<='):
            return interval(NegativeInfinity(), root, True, op == '<')
        return None
    if set(coefficients) <= {0, degree} and degree % 2 == 0:
        # x**(2k) op c
        c = -coefficients.get(0, _ZERO)/lead
        if c.is_positive is not True:
            return None
        from sympy_extras.core.power import pow
        r = pow(c, Rational(1, degree))
        if op in ('<', '<='):
            return interval(-r, r, op == '<', op == '<')
        if op in ('>', '>='):
            return union(interval(NegativeInfinity(), -r, True, op == '>'), interval(r, oo, op == '>', True))
    return None


# ---------------------------------------------------------------------------
# the real relations

def _algebraic(f: Expr, x: Symbol) -> bool:
    """Whether ``x`` occurs in ``f`` only in sums, products and powers
    with exponents free of ``x``."""
    for a in f.atoms(Application):
        if a.has(x):
            return False
    for p in f.atoms(Pow):
        if isinstance(p, Pow) and isinstance(p.exp, Expr) and p.exp.has(x):
            return False
    return True


def _without_periodic(f: Expr, x: Symbol) -> bool:
    """Whether the functions of ``x`` in ``f`` are exponentials,
    logarithms and powers only (their real solutions are the principal
    ones)."""
    from sympy_extras.functions.elementary.exponential import LambertW, Log
    for a in f.atoms(Application):
        if a.has(x) and not isinstance(a, (Exp, Log, LambertW)):
            return False
    return True


def _candidates_real(f: Expr, x: Symbol) -> Optional[Set]:
    """The real solutions of ``f = 0`` from the candidates of
    :func:`~sympy_extras.solvers.solvers.solve_equation`: the real ones
    which do not provably fail; ``None`` when it has no method, or a
    candidate cannot be decided."""
    from .solvers import checksol, solve_equation
    found = attempt(lambda: solve_equation(f, x), settings.timeout, native=True)
    if found is None:
        return None
    kept: list[Expr] = []
    for v in found:
        real = v.is_real
        if real is None:
            value = _numeric(v)
            if value is None:
                return None
            if abs(value.imag) > 10.0**(-(settings.precision//2))*max(1.0, abs(value)):
                real = False
            else:
                return None
        if real is False:
            continue
        verdict = checksol(f, x, v)
        if verdict is False:
            continue
        if verdict is None:
            return None
        kept.append(v)
    return finite_set(*kept) if kept else EmptySet


def _formula_solver(formula: Boolean, v: Symbol) -> Optional[Set]:
    """The solver given to the transcendental solver: a Boolean
    combination of relations polynomial in ``v``."""
    if isinstance(formula, BooleanTrue):
        return Reals
    if isinstance(formula, BooleanFalse):
        return EmptySet
    if isinstance(formula, Relational):
        found = _solve_real(formula, v)
        return found
    if isinstance(formula, (And, Or)):
        parts: list[Set] = []
        for a in formula.args:
            if not isinstance(a, Boolean):
                return None
            part = _formula_solver(a, v)
            if part is None:
                return None
            parts.append(part)
        return intersection(*parts) if isinstance(formula, And) else union(*parts)
    return None


def _solve_real(relation: Relational, x: Symbol) -> Optional[Set]:
    f = _expr(relation.lhs - relation.rhs)
    op = relation.rel_op
    if not f.has(x):
        return None
    from .solvers import _coefficients
    numerator, denominator = _fraction(f)
    rational = _coefficients(numerator, x) is not None and _coefficients(denominator, x) is not None
    if rational:
        if not (f.free_symbols - {x}):
            return real_polynomial_solutions(f, x, op)
        return _parametric_real(f, x, op)
    from .transcendental import solve_transcendental
    found = attempt(lambda: solve_transcendental(relation, x, _formula_solver), settings.timeout, native=True)
    if found is not None:
        return found
    if op == '==' and _without_periodic(f, x):
        return _candidates_real(f, x)
    if op == '==':
        return _periodic_real(f, x)
    return None


# ---------------------------------------------------------------------------
# the complex equations

#: the inverse images of a function over the complex numbers: the base
#: values and the period
Branches = list[tuple[Expr, Expr]]


def _complex_branches(g: Expr, r: Expr) -> Optional[Branches]:
    """The solutions ``u`` of ``g(u) = r`` as ``u = v + n*period`` (SymPy's
    ``_invert_complex`` and ``_solve_trig``)."""
    from sympy_extras.functions.elementary import hyperbolic as h
    from sympy_extras.functions.elementary import trigonometric as t
    if isinstance(g, Exp):
        if r == _ZERO:
            return []
        return [(log(r), 2*pi*I)]
    if isinstance(g, t.Sin):
        a = t.asin(r)
        return [(a, 2*pi), (pi - a, 2*pi)]
    if isinstance(g, t.Cos):
        a = t.acos(r)
        return [(a, 2*pi), (2*pi - a, 2*pi)]
    if isinstance(g, t.Tan):
        return [(t.atan(r), pi)]
    if isinstance(g, h.Sinh):
        a = h.asinh(r)
        return [(a, 2*pi*I), (I*pi - a, 2*pi*I)]
    if isinstance(g, h.Cosh):
        a = h.acosh(r)
        return [(a, 2*pi*I), (2*pi*I - a, 2*pi*I)]
    if isinstance(g, h.Tanh):
        return [(h.atanh(r), pi*I)]
    return None


def _peel(f: Expr, x: Symbol) -> tuple[Expr, Expr]:
    """``(g, r)`` with ``f = 0`` equivalent to ``g = r``, the terms and
    factors free of ``x`` moved to the right."""
    lhs: Expr = f
    r: Expr = _ZERO
    for _ in range(50):
        if isinstance(lhs, Add):
            independent, dependent = lhs.as_independent(x, as_Add=True)
            if independent == _ZERO:
                break
            lhs, r = dependent, r - independent
            continue
        if isinstance(lhs, Mul):
            independent, dependent = lhs.as_independent(x, as_Add=False)
            if independent == _ONE:
                break
            lhs, r = dependent, r/independent
            continue
        break
    return lhs, r


def _periodic(f: Expr, x: Symbol) -> Optional[Set]:
    """``g(u(x)) = c`` for an exponential, trigonometric or hyperbolic
    ``g``: the images of the integers, ``u(x) = v + n*period`` solved for
    ``x`` (by a division for a linear ``u``)."""
    from sympy_extras.polys.polytools import cancel
    from .solvers import solve_equation
    g, r = _peel(f, x)
    if not isinstance(g, Application) or len(g.args) != 1 or r.has(x):
        return None
    u = _expr(g.args[0])
    branches = _complex_branches(g, r)
    if branches is None:
        return None
    n = Dummy('n', integer=True)
    a = diff(u, x)
    linear = not a.has(x) and a != _ZERO
    b = _expr(u.subs(x, _ZERO)) if linear else _ZERO
    images: list[Set] = []
    seen: list[Expr] = []
    for value, period in branches:
        expressions: list[Expr]
        if linear:
            expressions = [cancel((value + n*period - b)/a)]
        else:
            target = value + n*period
            found = attempt(lambda: solve_equation(u - target, x), settings.timeout, native=True)
            if found is None:
                return None
            expressions = found
        for expression in expressions:
            if any(cancel(expression - other) == _ZERO for other in seen):
                continue
            seen.append(expression)
            images.append(image_set(Tuple(n), expression, Integers))
    return union(*images) if images else EmptySet


def _periodic_real(f: Expr, x: Symbol) -> Optional[Set]:
    """``g(u(x)) = c`` over the reals for a sine, a cosine or a tangent of
    a real argument and a real ``c`` (``|c| <= 1`` for the sine and the
    cosine): the images of the integers which are real."""
    from sympy_extras.functions.elementary import trigonometric as t
    g, r = _peel(f, x)
    if not isinstance(g, (t.Sin, t.Cos, t.Tan)) or r.has(x) or r.is_real is not True:
        return None
    if not isinstance(g, t.Tan):
        from sympy_extras.functions.elementary.complexes import abs_
        if not isinstance(abs_(r) <= 1, BooleanTrue):
            return None
    found = _periodic(f, x)
    if found is None:
        return None
    from sympy_extras.sets.fancysets import ImageSet
    from sympy_extras.sets.sets import Union as SetUnion
    pieces = found.sets if isinstance(found, SetUnion) else (found,)
    for piece in pieces:
        if not isinstance(piece, ImageSet):
            return None
        image = piece.image
        # real for the integers at which it is defined (1/(2*n) at n = 0 is
        # not a number, and SymPy keeps such images too)
        if not isinstance(image, Expr) or image.is_real is False:
            return None
    return found


def _solve_complex(f: Expr, x: Symbol) -> Optional[Set]:
    from .solvers import _coefficients, _polynomial_roots, checksol, solve_equation, zero_truth
    numerator, denominator = _fraction(f)
    coefficients = _coefficients(numerator, x)
    if coefficients is not None and not any(c.has(x) for c in coefficients.values()) \
            and _coefficients(denominator, x) is not None:
        degree = max(coefficients) if coefficients else 0
        if degree == 0:
            return EmptySet
        found = _polynomial_roots(numerator, x, degree)
        if found is None:
            found = attempt(lambda: solve_equation(numerator, x), settings.timeout, native=True)
        if found is not None:
            found = _expanded_numbers(found)
            kept = [r for r in found if not denominator.has(x) or zero_truth(_expr(denominator.subs(x, r))) is not True]
            return finite_set(*kept) if kept else EmptySet
        return None
    periodic = _periodic(f, x)
    if periodic is not None:
        return periodic
    if _algebraic(f, x):
        candidates = attempt(lambda: solve_equation(f, x), settings.timeout, native=True)
        if candidates is None:
            return None
        kept_: list[Expr] = []
        for v in candidates:
            verdict = checksol(f, x, v)
            if verdict is False:
                continue
            if verdict is None:
                return None
            kept_.append(v)
        return finite_set(*kept_) if kept_ else EmptySet
    return None


# ---------------------------------------------------------------------------
# the entry point

def solveset(relation: Boolean, x: Symbol, domain: Set) -> Set:
    """The solutions of a relation in ``x`` in ``domain`` (SymPy's
    ``solveset(relation, x, domain)``): a set of the core, the
    ``ConditionSet`` of the relation when it is not solved (see the
    module).

    >>> from sympy_extras.core import Eq, Symbol, I
    >>> from sympy_extras.functions import sin, log
    >>> from sympy_extras.sets import Reals, Complexes
    >>> from sympy_extras.solvers.solveset import solveset
    >>> x, a = Symbol('x'), Symbol('a')
    >>> solveset(Eq(2*x - I, 0), x, Reals), solveset(Eq(2*x - I, 0), x, Complexes)
    (EmptySet, {I/2})
    >>> solveset(Eq(sin(a*x), 0), x, Complexes)
    Union(ImageSet(Lambda(_n, (2*_n*pi + pi)/a), Integers), ImageSet(Lambda(_n, 2*_n*pi/a), Integers))
    >>> solveset(Eq(log(x) + log(x - 1), log(2)), x, Reals)
    {2}
    """
    if isinstance(relation, BooleanTrue):
        return domain
    if isinstance(relation, BooleanFalse):
        return EmptySet
    unsolved = condition_set(x, relation, domain)
    if not isinstance(relation, Relational) or not relation.has(x):
        return unsolved
    real = isinstance(domain, RealsClass) or domain.is_subset(Reals) is True
    if real:
        found = _solve_real(relation, x)
        if found is None:
            return unsolved
        return found if isinstance(domain, RealsClass) else intersection(found, domain)
    if relation.rel_op != '==':
        return unsolved
    found = _solve_complex(_expr(relation.lhs - relation.rhs), x)
    if found is None:
        return unsolved
    return found if isinstance(domain, ComplexesClass) else intersection(found, domain)

