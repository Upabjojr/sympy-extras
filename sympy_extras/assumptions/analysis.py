"""Signs of expressions of one real variable which are not polynomials.

The cylindrical algebraic decomposition decides statements about
polynomials. For an expression `f(x)` with exponentials, logarithms,
trigonometric or other elementary functions, the sign of `f` on the set
where `x` is assumed to lie is found from calculus:

1. `f` must be continuous on the set: every function applied is
   continuous where its argument lies (the functions of SymPy's
   ``continuous_domain``, checked on the signs of their arguments);
2. the zeros of `f` on the set are computed exactly (the rational
   functions by :func:`~sympy_extras.polys.rationalsolve.solveset_rational`,
   the others factor by factor with the transcendental solver
   :func:`~sympy_extras.solvers.transcendental.solve_transcendental`, the
   periodic families listed on bounded sets);
   between consecutive zeros (and endpoints) the sign is constant, so it
   is read off at one rational sample point per piece, with exact
   evaluation or, when :data:`sympy_extras.settings.settings` allows it,
   certified numerical evaluation (rigorous interval arithmetic at the
   precision of the settings, or for the functions it does not cover a
   value computed with a margin over the precision before its sign is
   trusted);
3. when the zeros cannot be found, interval arithmetic (accumulation
   bounds, :mod:`sympy_extras.core.accumbounds`) bounds `f` on the set,
   the critical points of `f'` split the set into monotone pieces whose
   signs follow from the limits at the ends, and finally the zeros are
   isolated by bisection with rigorous interval arithmetic
   (:mod:`sympy_extras.assumptions.intervals`).

Expressions of several variables are handled on the box of their
assumptions by branch and bound with interval arithmetic, by monotonicity
in every variable, or through the range of a single inner argument.

Every answer is a proof modulo the correctness of the solvers; the
undecided cases give ``None``.

The module works on the expressions of the core and imports no SymPy:
:mod:`sympy_extras.assumptions.analysis_sympy` is its face on SymPy
objects. No step goes through SymPy at run time: the intersection of the
solutions of a polynomial assumption which is not a bound (the native
CAD's ``solution_set``) with other intervals, when the core does not
compare their ends, is made by certified comparisons.
"""
from __future__ import annotations

from typing import Optional

from sympy_extras.core.accumbounds import AccumBounds, accum_bounds
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.function import count_ops
from sympy_extras.core.limits import limit
from sympy_extras.core.logic import And, Boolean, Or, false, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import (Float, I, ImaginaryUnit, Infinity, Integer, NegativeInfinity, Rational,
    number_from_fraction, oo)
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.hyperbolic import (Acosh, Acoth, Asech, Asinh, Atanh, Cosh, Coth, Csch, Sech,
    Sinh, Tanh)
from sympy_extras.functions.elementary.trigonometric import (Acos, Acot, Acsc, Asec, Asin, Atan, Cos, Cot, Csc, Sec,
    Sin, Tan, cos, sin)
from sympy_extras.polys.rationalsolve import solveset_rational
from sympy_extras.sets.fancysets import ImageSet, IntegersClass, Naturals0Class, NaturalsClass
from sympy_extras.sets.sets import (Contains, EmptySet, EmptySetClass, FiniteSet, Intersection, Interval, Reals, Set,
    Union, finite_set, intersection, interval, union)
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

from .facts import Facts, to_polynomial
from .intervals import (Box, Openness, Sign, Signs, corner_signs, evaluate, interval_sign, isolate_signs,
    monotone_range, range_on_box, replaced, signs_on_box)

__all__ = ['Truth', 'certified_sign', 'certified_compare', 'domain_of', 'sign_on', 'signs_multivariate', 'box_of',
    'decide_relational']

#: a truth value of three states
Truth = Optional[bool]


def _symbols(e: Basic) -> set[Symbol]:
    """The free symbols which are plain symbols."""
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


def _interval_sign_at(value: Expr, digits: int) -> tuple[bool, Optional[Sign]]:
    """The sign of a constant by interval arithmetic at ``digits`` digits:
    whether the arithmetic covers it, and the sign when the enclosure
    excludes zero."""
    import mpmath
    saved = mpmath.iv.prec
    mpmath.iv.dps = digits
    try:
        enclosure = evaluate(value, {})
    finally:
        mpmath.iv.prec = saved
    if enclosure is None:
        return False, None
    s = interval_sign(enclosure)
    return True, (s if s != 0 else None)


def _is_algebraic_constant(e: Expr) -> bool:
    """Whether ``e`` is built from rationals and ``I`` by sums, products
    and rational powers."""
    if isinstance(e, (Rational, ImaginaryUnit)):
        return True
    if isinstance(e, Pow):
        return isinstance(e.exp, Rational) and _is_algebraic_constant(e.base)
    if isinstance(e, (Add, Mul)):
        return all(isinstance(a, Expr) and _is_algebraic_constant(a) for a in e.args)
    return False


def _vanishes(e: Expr) -> bool:
    """Whether the algebraic number ``e`` is zero: its minimal polynomial
    is ``x``."""
    from sympy_extras.polys.numberfields.minpoly import minimal_polynomial_coeffs
    coefficients = attempt(lambda: minimal_polynomial_coeffs(e), settings.timeout, native=True)
    return coefficients == [1, 0]


def certified_sign(value: Expr) -> Optional[Sign]:
    """The sign of a constant expression: from the exact knowledge of the
    core, or numerically when the numerical checks are enabled (rigorous
    interval arithmetic where it covers the expression, otherwise a value
    computed with a margin over its precision); ``None`` when undecided.

    Examples
    ========

    >>> from sympy_extras.core import E, Rational, Symbol
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.assumptions.analysis import certified_sign
    >>> certified_sign(E - 2), certified_sign(sin(3) - Rational(1, 7)), certified_sign(Symbol('x'))
    (1, -1, None)
    """
    if _symbols(value):
        return None
    if isinstance(value, Infinity):
        return 1
    if isinstance(value, NegativeInfinity):
        return -1
    if value.is_zero:
        return 0
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    if _is_algebraic_constant(value) and _vanishes(value):
        # SymPy knows a hidden zero of an algebraic number from its
        # minimal polynomial (Expr._eval_is_extended_positive_negative)
        return 0
    if not settings.numerical_checks or value.is_real is False:
        return None
    for digits in (settings.precision, 2*settings.precision, 4*settings.precision):
        covered, s = _interval_sign_at(value, digits)
        if covered:
            # an enclosure is a proof; one containing zero says nothing
            if s is not None:
                return s
            continue
        approximation = value.evalf(digits)
        if not isinstance(approximation, Float) or not approximation.is_finite:
            return None
        # a value which is not zero to that precision has a certain sign
        threshold = Float(10)**Integer(-(digits//2))
        if isinstance(threshold, Float) and abs(approximation.as_fraction()) > threshold.as_fraction():
            return 1 if approximation.as_fraction() > 0 else -1
    return None


def certified_compare(a: Expr, b: Expr) -> Optional[Sign]:
    """The sign of ``a - b`` for constants, infinities included.

    Examples
    ========

    >>> from sympy_extras.core import pi, oo, Integer
    >>> from sympy_extras.assumptions.analysis import certified_compare
    >>> certified_compare(pi, Integer(3)), certified_compare(oo, oo), certified_compare(Integer(2), -oo)
    (1, 0, 1)
    """
    if isinstance(a, Infinity) or isinstance(b, NegativeInfinity):
        return 0 if a == b else 1
    if isinstance(a, NegativeInfinity) or isinstance(b, Infinity):
        return 0 if a == b else -1
    return certified_sign(a - b)


def _interval_of_bound(atom: Boolean, x: Symbol) -> Optional[Set]:
    """The interval described by a bound ``x rel c`` with a constant
    ``c``."""
    if not isinstance(atom, Relational) or _symbols(atom) != {x}:
        return None
    lhs, rhs = atom.lhs, atom.rhs
    if lhs == x and not _symbols(rhs):
        c, flipped = rhs, False
    elif rhs == x and not _symbols(lhs):
        c, flipped = lhs, True
    else:
        return None
    if not c.is_real:
        return None
    kind: type[Relational] = type(atom)
    if flipped:
        flip: dict[type[Relational], type[Relational]] = {Gt: Lt, Lt: Gt, Ge: Le, Le: Ge}
        kind = flip.get(kind, kind)
    if kind is Gt:
        return interval(c, oo, True, True)
    if kind is Ge:
        return interval(c, oo)
    if kind is Lt:
        return interval(-oo, c, True, True)
    if kind is Le:
        return interval(-oo, c)
    if kind is Eq:
        from sympy_extras.sets.sets import finite_set
        return finite_set(c)
    if kind is Ne:
        return union(interval(-oo, c, True, True), interval(c, oo, True, True))
    return None


def domain_of(x: Symbol, facts: Facts) -> Optional[Set]:
    """The set of real values of ``x`` allowed by the facts which only
    concern ``x``, as a union of intervals; ``None`` when some fact about
    ``x`` involves other symbols or cannot be turned into intervals.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.assumptions.facts import Facts
    >>> from sympy_extras.assumptions.analysis import domain_of
    >>> x, y = Symbol('x'), Symbol('y')
    >>> domain_of(x, Facts([x > 0, x < pi])), domain_of(x, Facts([x > y]))
    (Interval.open(0, pi), None)
    """
    if x not in facts.real:
        return None
    domain: Set = Reals
    for atom in facts.conjuncts:
        symbols = _symbols(atom)
        if x not in symbols:
            continue
        if symbols != {x}:
            return None
        piece: Optional[Set] = None
        if isinstance(atom, Contains):
            candidate = atom.args[1]
            if isinstance(candidate, Interval):
                piece = candidate
        if piece is None:
            piece = _interval_of_bound(atom, x)
        on_roots = False
        if piece is None and to_polynomial(atom, facts.real) is not None:
            # the CAD describes the solutions of the polynomial
            piece = _cad_solution_set(atom, x)
            on_roots = True
        if piece is None:
            return None
        found = intersection(domain, piece)
        if on_roots and not _is_union_of_intervals(found):
            # ends which the core does not compare (roots of polynomials
            # against other numbers): by certified comparisons
            certified = _certified_intersection(domain, piece)
            if certified is not None:
                found = certified
        domain = found
    if isinstance(domain, (EmptySetClass, Interval)):
        return domain
    if isinstance(domain, Union) and all(isinstance(a, Interval) for a in domain.args):
        return domain
    return None


def _cad_solution_set(atom: Boolean, x: Symbol) -> Optional[Set]:
    """The solutions of a polynomial relation in ``x`` by the CAD
    (:func:`sympy_extras.polys.cad.qe.solution_set`), ``None`` when it
    fails or does not end in time."""
    from sympy_extras.polys.cad.qe import solution_set
    return attempt(lambda: solution_set(atom, x), settings.timeout, native=True)


def _intersect_pieces(a: Interval, b: Interval) -> Optional[Set]:
    """The intersection of two intervals whose ends are compared by
    :func:`certified_compare` (``None`` when two ends cannot be told
    apart)."""
    starts = certified_compare(a.start, b.start)
    ends = certified_compare(a.end, b.end)
    if starts is None or ends is None:
        return None
    if starts > 0 or (starts == 0 and a.left_open):
        start, left_open = a.start, a.left_open
    else:
        start, left_open = b.start, b.left_open
    if ends < 0 or (ends == 0 and a.right_open):
        end, right_open = a.end, a.right_open
    else:
        end, right_open = b.end, b.right_open
    order = certified_compare(start, end)
    if order is None:
        return None
    if order > 0:
        return EmptySet
    if order == 0:
        return EmptySet if left_open or right_open else finite_set(start)
    return Interval(start, end, left_open, right_open)


def _certified_intersection(a: Set, b: Set) -> Optional[Set]:
    """The intersection of two unions of intervals, their ends compared by
    certified numerical comparisons (roots of polynomials against other
    constants), ``None`` when a comparison is undecided."""
    first, second = _pieces(a), _pieces(b)
    if first is None or second is None:
        return None
    parts: list[Set] = []
    for p in first:
        for q in second:
            part = _intersect_pieces(p, q)
            if part is None:
                return None
            if isinstance(part, EmptySetClass):
                continue
            if not isinstance(part, Interval):
                # an isolated point: not a union of intervals
                return None
            parts.append(part)
    if not parts:
        return EmptySet
    return union(*parts)


def _is_union_of_intervals(s: Set) -> bool:
    return isinstance(s, (EmptySetClass, Interval)) or (isinstance(s, Union) and
                                                        all(isinstance(a, Interval) for a in s.args))


def _pieces(domain: Set) -> Optional[list[Interval]]:
    if domain is Reals:
        return [Interval(-oo, oo, True, True)]
    if isinstance(domain, Interval):
        return [domain]
    if isinstance(domain, Union):
        pieces: list[Interval] = []
        for a in domain.args:
            if not isinstance(a, Interval):
                return None
            pieces.append(a)
        return pieces
    return None


def _rational(value: Expr) -> Expr:
    """The exact value of a float (``Rational(Float)`` of SymPy)."""
    if isinstance(value, Float):
        return number_from_fraction(value.as_fraction())
    if isinstance(value, Rational):
        return value
    raise TypeError("invalid input: %s" % (value,))


def _between(p: Expr, q: Expr) -> Optional[Expr]:
    """A rational number strictly between the constants ``p < q``."""
    if isinstance(p, NegativeInfinity) and isinstance(q, Infinity):
        return Integer(0)
    for digits in (settings.precision, 2*settings.precision):
        if isinstance(p, NegativeInfinity):
            candidate = _rational(q.evalf(digits)) - 1
        elif isinstance(q, Infinity):
            candidate = _rational(p.evalf(digits)) + 1
        else:
            candidate = _rational((p.evalf(digits) + q.evalf(digits))/2)
        if certified_compare(candidate, p) == 1 and certified_compare(q, candidate) == 1:
            return candidate
    return None


def _sorted_constants(values: list[Expr]) -> Optional[list[Expr]]:
    """The constants sorted by certified comparisons (``None`` if two of
    them cannot be told apart)."""
    result: list[Expr] = []
    for v in values:
        position = len(result)
        for i, w in enumerate(result):
            c = certified_compare(v, w)
            if c is None:
                return None
            if c == 0:
                position = -1
                break
            if c < 0:
                position = i
                break
        if position >= 0:
            result.insert(position, v)
    return result


# ---------------------------------------------------------------------------
# continuity and zeros: native for the rational functions, SymPy otherwise

def _is_rational_in(f: Expr, x: Symbol) -> bool:
    """Whether ``f`` is a rational function of ``x`` (its subexpressions
    free of ``x`` are constants)."""
    if x not in f.free_symbols:
        return True
    if f == x:
        return True
    if isinstance(f, Pow):
        return isinstance(f.exp, Integer) and _is_rational_in(f.base, x)
    if isinstance(f, (Add, Mul)):
        return all(isinstance(a, Expr) and _is_rational_in(a, x) for a in f.args)
    return False


def _real_zeros_in(f: Expr, x: Symbol, domain: Set) -> tuple[bool, list[Expr]]:
    """The zeros of a rational ``f`` in ``domain`` natively: whether they
    are known (every complex zero decided in or out of the domain), and
    the list."""
    solutions = solveset_rational(f, x)
    if solutions is None:
        return False, []
    zeros: list[Expr] = []
    for z in solutions[0]:
        member = domain.contains(z)
        if member is true:
            zeros.append(z)
        elif member is not false:
            return False, []
    return True, zeros


#: the functions continuous on the whole real line (of a continuous argument)
_ENTIRE: tuple[type[Basic], ...] = (Exp, Sin, Cos, Atan, Sinh, Cosh, Tanh, Sech, Asinh, Abs)
#: the functions continuous where their argument is not zero
_POLE_AT_ZERO: tuple[type[Basic], ...] = (Coth, Csch, Acot)


def _signs_within(f: Expr, x: Symbol, domain: Set, allowed: Signs) -> bool:
    """Whether every sign of ``f`` on ``domain`` is among ``allowed``."""
    signs = sign_on(f, x, domain)
    return signs is not None and signs <= allowed


def _either_side(f: Expr, low: Expr, high: Expr, strict: bool, x: Symbol, domain: Set) -> bool:
    """Whether on every piece of ``domain`` the values of ``f`` stay on one
    side: ``f <= low`` or ``f >= high`` (strictly with ``strict``)."""
    pieces = _pieces(domain)
    if pieces is None:
        return False
    below: Signs = frozenset([-1]) if strict else frozenset([-1, 0])
    above: Signs = frozenset([1]) if strict else frozenset([0, 1])
    for piece in pieces:
        if not (_signs_within(f - low, x, piece, below) or _signs_within(f - high, x, piece, above)):
            return False
    return True


def _continuous_on(f: Expr, x: Symbol, domain: Set) -> bool:
    """Whether ``f`` is continuous on ``domain`` (a union of intervals):
    every function applied is continuous where its argument lies, as in
    SymPy's ``continuous_domain`` (the functions it implements: powers,
    ``exp``, ``log``, ``Abs``, the trigonometric and hyperbolic functions
    and their inverses), checked natively on the subexpressions by the
    signs of their arguments; ``False`` when this cannot be established
    (another function, an undecided sign)."""
    if x not in f.free_symbols or f == x:
        return True
    if isinstance(f, (Add, Mul)):
        return all(isinstance(a, Expr) and _continuous_on(a, x, domain) for a in f.args)
    if isinstance(f, Pow):
        return _power_continuous_on(f, x, domain)
    if len(f.args) != 1:
        return False
    argument = f.args[0]
    if not isinstance(argument, Expr) or not _continuous_on(argument, x, domain):
        return False
    if isinstance(f, _ENTIRE):
        return True
    if isinstance(f, _POLE_AT_ZERO):
        return _signs_within(argument, x, domain, frozenset([-1, 1]))
    if isinstance(f, (Tan, Sec)):
        return _signs_within(cos(argument), x, domain, frozenset([-1, 1]))
    if isinstance(f, (Cot, Csc)):
        return _signs_within(sin(argument), x, domain, frozenset([-1, 1]))
    if isinstance(f, Log):
        return _signs_within(argument, x, domain, frozenset([1]))
    if isinstance(f, (Asin, Acos)):
        return _signs_within(argument - 1, x, domain, frozenset([-1, 0])) and \
            _signs_within(argument + 1, x, domain, frozenset([0, 1]))
    if isinstance(f, Acosh):
        return _signs_within(argument - 1, x, domain, frozenset([0, 1]))
    if isinstance(f, Atanh):
        return _signs_within(argument - 1, x, domain, frozenset([-1])) and \
            _signs_within(argument + 1, x, domain, frozenset([1]))
    if isinstance(f, Asech):
        return _signs_within(argument, x, domain, frozenset([1])) and \
            _signs_within(argument - 1, x, domain, frozenset([-1, 0]))
    if isinstance(f, (Asec, Acsc)):
        return _either_side(argument, Integer(-1), Integer(1), False, x, domain)
    if isinstance(f, Acoth):
        return _either_side(argument, Integer(-1), Integer(1), True, x, domain)
    return False


def _power_continuous_on(f: Pow, x: Symbol, domain: Set) -> bool:
    """A power ``b**e`` on ``domain``: an integer or odd-root exponent
    needs a nonzero base when negative; another constant exponent a base
    which is not negative (positive for a negative exponent); an exponent
    in ``x`` a positive base."""
    base, exponent = f.base, f.exp
    if x not in exponent.free_symbols:
        if not _continuous_on(base, x, domain):
            return False
        if isinstance(exponent, Integer):
            if exponent.p >= 0:
                return True
            if _is_rational_in(base, x):
                found, poles = _real_zeros_in(base, x, domain)
                if found:
                    return not poles
            return _signs_within(base, x, domain, frozenset([-1, 1]))
        if isinstance(exponent, Rational) and exponent.q % 2 == 1:
            return exponent.p > 0 or _signs_within(base, x, domain, frozenset([-1, 1]))
        if exponent.is_positive is True:
            return _signs_within(base, x, domain, frozenset([0, 1]))
        return _signs_within(base, x, domain, frozenset([1]))
    if not _continuous_on(exponent, x, domain):
        return False
    if x not in base.free_symbols:
        if exponent.is_integer is True:
            # an integer variable (as SymPy's continuous_domain, whose
            # rational exponents have no constraint): (-1)**k
            return certified_sign(base) in (-1, 1)
        return certified_sign(base) == 1
    return _continuous_on(base, x, domain) and _signs_within(base, x, domain, frozenset([1]))


def _native_solver(formula: Boolean, v: Symbol) -> Optional[Set]:
    """The solver of the relations polynomial in a kernel variable with
    constant coefficients which :func:`solve_transcendental` is given: the
    linear relations natively, ``None`` for the others."""
    from sympy_extras.solvers.transcendental import _linear_solution
    if isinstance(formula, Relational):
        return _linear_solution(formula, v)
    if isinstance(formula, (And, Or)):
        parts: list[Set] = []
        for a in formula.args:
            if not isinstance(a, Boolean):
                return None
            part = _native_solver(a, v)
            if part is None:
                return None
            parts.append(part)
        return intersection(*parts) if isinstance(formula, And) else union(*parts)
    return None


def _zero_set(f: Expr, x: Symbol) -> Optional[Set]:
    """The real zeros of ``f`` by the native
    :func:`~sympy_extras.solvers.transcendental.solve_transcendental`
    (no step on SymPy: the linear relations of its kernels natively, no
    Lambert W candidates), factor by factor of its common factors as
    SymPy's ``solveset`` does (``exp(x)*(sin(x) + cos(x))``; the callers
    have checked that every factor is finite); ``None`` when it fails."""
    from sympy_extras.solvers.transcendental import solve_transcendental
    g = factor_terms(f)
    if isinstance(g, Mul):
        parts: list[Set] = []
        for a in g.args:
            if not isinstance(a, Expr) or x not in a.free_symbols:
                continue
            part = _zero_set(a, x)
            if part is None:
                return None
            parts.append(part)
        return union(*parts) if parts else EmptySet
    if isinstance(g, Pow) and x not in g.exp.free_symbols:
        if g.exp.is_positive is True:
            return _zero_set(g.base, x)
        if g.exp.is_negative is True:
            return EmptySet
    if isinstance(g, Exp):
        return EmptySet
    return solve_transcendental(Eq(g, Integer(0)), x, solver=_native_solver, lambert=False)


#: the most points of a periodic family of zeros listed in a bounded piece
_MAX_PERIODIC_POINTS = 1000


def _image_points(image: Expr, k: Symbol, piece: Interval) -> Optional[list[Expr]]:
    """The points ``a*k + b`` (``k`` an integer, ``a`` and ``b`` real
    constants) in a bounded piece, ``None`` when the image is not linear
    or the piece is unbounded."""
    a = image.diff(k)
    b = as_expr(image.subs(k, Integer(0)))
    if a.has(k) or _symbols(a) or _symbols(b):
        return None
    sa = certified_sign(a)
    if sa is None:
        return None
    if sa == 0:
        member = piece.contains(b)
        if member is true:
            return [b]
        return [] if member is false else None
    if not (piece.start.is_finite and piece.end.is_finite):
        return None
    ends: list[float] = []
    for end in (piece.start, piece.end):
        value = ((end - b)/a).evalf(settings.precision)
        if not isinstance(value, (Float, Rational)):
            return None
        ends.append(float(value))
    low, high = int(min(ends)) - 2, int(max(ends)) + 2
    if high - low > _MAX_PERIODIC_POINTS:
        return None
    points: list[Expr] = []
    for j in range(low, high + 1):
        point = as_expr(image.subs(k, Integer(j)))
        above = certified_compare(point, piece.start)
        below = certified_compare(piece.end, point)
        if above is None or below is None:
            return None
        if (above > 0 or (above == 0 and not piece.left_open)) and \
                (below > 0 or (below == 0 and not piece.right_open)):
            points.append(point)
    return points


def _points_in(zeros: Set, domain: Set) -> Optional[list[Expr]]:
    """The points of a set of zeros in ``domain`` when they are finitely
    many and each is decided in or out (``None`` otherwise)."""
    if isinstance(zeros, EmptySetClass):
        return []
    if isinstance(zeros, FiniteSet):
        points: list[Expr] = []
        for z in zeros.args:
            if not isinstance(z, Expr):
                return None
            member = domain.contains(z)
            if member is true:
                points.append(z)
            elif member is not false:
                return None
        return points
    if isinstance(zeros, Union):
        collected: list[Expr] = []
        for part in zeros.args:
            if not isinstance(part, Set):
                return None
            found = _points_in(part, domain)
            if found is None:
                return None
            collected.extend(z for z in found if z not in collected)
        return collected
    if isinstance(zeros, ImageSet):
        variables, image = zeros.variables, zeros.image
        if len(variables) != 1 or len(zeros.base_sets) != 1 or not isinstance(zeros.base_sets[0], IntegersClass) \
                or not isinstance(image, Expr):
            return None
        pieces = _pieces(domain)
        if pieces is None:
            return None
        listed: list[Expr] = []
        for piece in pieces:
            part_points = _image_points(image, variables[0], piece)
            if part_points is None:
                return None
            listed.extend(z for z in part_points if z not in listed)
        return listed
    return None


def _is_discrete(zeros: Set, piece: Interval) -> bool:
    """Whether a set of zeros meets the piece in isolated points: finite
    sets, images of the integers, intervals meeting the piece in at most
    one point, and their unions."""
    if isinstance(zeros, (FiniteSet, EmptySetClass)):
        return True
    if isinstance(zeros, ImageSet):
        return len(zeros.base_sets) == 1 and isinstance(zeros.base_sets[0], (IntegersClass, NaturalsClass,
                                                                              Naturals0Class))
    if isinstance(zeros, Interval):
        return isinstance(intersection(zeros, piece), (EmptySetClass, FiniteSet))
    if isinstance(zeros, Union):
        return all(isinstance(a, Set) and _is_discrete(a, piece) for a in zeros.args)
    if isinstance(zeros, Intersection):
        return any(isinstance(a, Set) and _is_discrete(a, piece) for a in zeros.args)
    return False


def _finite_zeros(f: Expr, x: Symbol, domain: Set) -> Optional[list[Expr]]:
    """The zeros of ``f`` in ``domain`` when they are finitely many and
    found exactly, ``None`` otherwise: a rational ``f`` by
    :func:`~sympy_extras.polys.rationalsolve.solveset_rational`, another
    one by the native transcendental solver."""
    if _is_rational_in(f, x):
        found, zeros = _real_zeros_in(f, x, domain)
        if found:
            return zeros
    solutions = _zero_set(f, x)
    if solutions is None:
        return None
    return _points_in(solutions, domain)


def _has_discrete_zeros(f: Expr, x: Symbol, piece: Interval) -> bool:
    """Whether the zeros of ``f`` on the piece are isolated points (a
    finite set, or a periodic family)."""
    if _is_rational_in(f, x) and _real_zeros_in(f, x, piece)[0]:
        return True
    solutions = _zero_set(f, x)
    return solutions is not None and _is_discrete(solutions, piece)


def sign_on(f: Expr, x: Symbol, domain: Set, depth: int = 1) -> Optional[Signs]:
    """The set of signs which ``f`` takes on ``domain`` (a union of
    intervals), or ``None`` when it cannot be established.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.sets import Interval, Reals
    >>> from sympy_extras.assumptions.analysis import sign_on
    >>> x = Symbol('x')
    >>> sorted(sign_on(x**2 - 2, x, Interval(0, 10, True, True))), sign_on(x**2 + 1, x, Reals)
    ([-1, 0, 1], frozenset({1}))
    >>> sign_on(1/x, x, Interval(-1, 1, True, True)) is None
    True
    """
    pieces = _pieces(domain)
    if pieces is None:
        return None
    if not f.has(x):
        s = certified_sign(f)
        return None if s is None else frozenset([s])
    continuous = attempt(lambda: _continuous_on(f, x, domain), settings.timeout, native=True)
    if continuous is not True:
        return None
    zero_list = attempt(lambda: _finite_zeros(f, x, domain), settings.timeout, native=True)
    if zero_list is not None and all(z.is_real for z in zero_list):
        return _signs_from_zeros(f, x, pieces, zero_list)
    bounded = _signs_from_bounds(f, x, pieces)
    if bounded is not None and len(bounded) == 1:
        return bounded
    if depth > 0:
        monotone = _signs_from_monotonicity(f, x, pieces, depth)
        if monotone is not None:
            return monotone
    # certified root isolation by interval arithmetic
    isolated: set[Sign] = set()
    for piece in pieces:
        part = isolate_signs(f, x, piece)
        if part is None:
            return bounded
        isolated.update(part)
    return frozenset(isolated)


def _signs_from_bounds(f: Expr, x: Symbol, pieces: list[Interval]) -> Optional[Signs]:
    """The signs allowed by interval arithmetic (accumulation bounds) on
    every piece: an over-approximation, so only a one-sided answer is
    informative."""
    signs: set[Sign] = set()
    for piece in pieces:
        bounds = attempt(lambda: f.subs(x, accum_bounds(piece.start, piece.end)), settings.timeout, native=True)
        if bounds is None:
            return None
        if isinstance(bounds, AccumBounds):
            low, high = bounds.min, bounds.max
        elif not _symbols(bounds):
            low = high = bounds
        else:
            return None
        low_sign, high_sign = certified_sign(low), certified_sign(high)
        if low_sign is None or high_sign is None:
            return None
        if low_sign == 1:
            signs.add(1)
        elif high_sign == -1:
            signs.add(-1)
        elif low_sign == 0 and high_sign == 0:
            signs.add(0)
        elif low_sign == 0:
            signs.update((0, 1))
        elif high_sign == 0:
            signs.update((-1, 0))
        else:
            signs.update((-1, 0, 1))
    return frozenset(signs)


def _signs_from_zeros(f: Expr, x: Symbol, pieces: list[Interval], zeros: list[Expr]) -> Optional[Signs]:
    signs: set[Sign] = set()
    if zeros:
        signs.add(0)
    for piece in pieces:
        a, b = piece.start, piece.end
        inside = [z for z in zeros if certified_compare(z, a) == 1 and certified_compare(b, z) == 1]
        ordered = _sorted_constants([a] + inside + [b])
        if ordered is None:
            return None
        if len(ordered) < 2:
            # a single point: its sign is the value there
            s = certified_sign(f.subs(x, ordered[0]))
            if s is None:
                return None
            signs.add(s)
            continue
        for p, q in zip(ordered, ordered[1:]):
            sample = _between(p, q)
            if sample is None:
                return None
            s = certified_sign(f.subs(x, sample))
            if s is None:
                return None
            signs.add(s)
    return frozenset(signs)


def _endpoint_value(f: Expr, x: Symbol, point: Expr, direction: str) -> Optional[Expr]:
    value = attempt(lambda: limit(f, x, point, direction), settings.timeout, native=True)
    if value is None or _symbols(value):
        return None
    return value


def _value_at(f: Expr, x: Symbol, point: Expr, direction: str) -> Optional[Expr]:
    """The value of a continuous ``f`` at a finite point, or its limit at
    an infinite one."""
    if point.is_finite:
        value = f.subs(x, point)
        if not _symbols(value) and value.is_finite is not False:
            return value
    return _endpoint_value(f, x, point, direction)


def _monotone_signs(f: Expr, x: Symbol, a: Expr, b: Expr, increasing: bool) -> Optional[set[Sign]]:
    """The signs of a strictly monotone ``f`` on the open interval
    ``(a, b)``, from its values or limits at the ends."""
    left = _value_at(f, x, a, '+')
    right = _value_at(f, x, b, '-')
    if left is None or right is None:
        return None
    low, high = (left, right) if increasing else (right, left)
    low_sign, high_sign = certified_sign(low), certified_sign(high)
    if low_sign is None or high_sign is None:
        return None
    if low_sign >= 0:
        return {1}
    if high_sign <= 0:
        return {-1}
    return {-1, 0, 1}


def _signs_from_monotonicity(f: Expr, x: Symbol, pieces: list[Interval], depth: int) -> Optional[Signs]:
    """The signs of ``f`` from its critical points: on every piece the
    zeros of ``f'`` are found; between them ``f`` is strictly monotone and
    its signs follow from its values at the ends. When the zeros of ``f'``
    cannot be listed, a constant sign of ``f'`` on the piece (with isolated
    zeros at most) is enough."""
    derivative = f.diff(x)
    signs: set[Sign] = set()
    for piece in pieces:
        a, b = piece.start, piece.end
        critical = attempt(lambda: _finite_zeros(derivative, x, piece), settings.timeout, native=True)
        if critical is not None:
            if not all(c.is_real for c in critical):
                return None
            inside = [c for c in critical if certified_compare(c, a) == 1 and certified_compare(b, c) == 1]
            ordered = _sorted_constants([a] + inside + [b])
            if ordered is None:
                return None
            for p, q in zip(ordered, ordered[1:]):
                sample = _between(p, q)
                if sample is None:
                    return None
                slope = certified_sign(derivative.subs(x, sample))
                if slope is None or slope == 0:
                    return None
                part = _monotone_signs(f, x, p, q, slope == 1)
                if part is None:
                    return None
                signs.update(part)
            for c in inside:
                value = certified_sign(f.subs(x, c))
                if value is None:
                    return None
                signs.add(value)
        else:
            slope_signs = sign_on(derivative, x, piece, depth - 1)
            if slope_signs is None or slope_signs == frozenset([0]) or (1 in slope_signs and -1 in slope_signs):
                return None
            strict = 0 not in slope_signs or _has_discrete_zeros(derivative, x, piece)
            if not strict:
                return None
            part = _monotone_signs(f, x, a, b, 1 in slope_signs)
            if part is None:
                return None
            signs.update(part)
        # closed finite ends: the value there, by continuity
        for closed, point, direction in ((not piece.left_open and a.is_finite, a, '+'),
                                         (not piece.right_open and b.is_finite, b, '-')):
            if closed:
                end_value = _value_at(f, x, point, direction)
                end_sign = certified_sign(end_value) if end_value is not None else None
                if end_sign is None:
                    return None
                signs.add(end_sign)
    return frozenset(signs)


def _numeric_key(e: Expr) -> tuple[int, Rational]:
    """The value of a real constant at 20 digits, as a sort key."""
    if isinstance(e, NegativeInfinity):
        return (-1, Integer(0))
    if isinstance(e, Infinity):
        return (1, Integer(0))
    value = e.evalf(20)
    if isinstance(value, Float):
        exact = number_from_fraction(value.as_fraction())
        return (0, exact)
    if isinstance(value, Rational):
        return (0, value)
    raise TypeError("Invalid comparison of non-real %s" % (value,))


def _key_less(a: tuple[int, Rational], b: tuple[int, Rational]) -> bool:
    if a[0] != b[0]:
        return a[0] < b[0]
    return a[1].as_fraction() < b[1].as_fraction()


def box_of(symbols: list[Symbol], facts: Facts) -> Optional[tuple[Box, Openness]]:
    """The box of the values of the symbols allowed by the facts which
    concern one symbol at a time (the hull of the pieces), and for every
    symbol whether the range is open at each end; ``None`` when a symbol
    is not known to be real.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.assumptions.facts import Facts
    >>> from sympy_extras.assumptions.analysis import box_of
    >>> x, y = Symbol('x'), Symbol('y')
    >>> box_of([x, y], Facts([x > 0, x < 1, y >= 2]))
    ({x: (0, 1), y: (2, oo)}, {x: (True, True), y: (False, True)})
    """
    box: Box = {}
    openness: Openness = {}
    for v in symbols:
        domain = domain_of(v, facts)
        if domain is None:
            return None
        pieces = _pieces(domain)
        if not pieces:
            return None
        first, last = pieces[0], pieces[0]
        first_key, last_key = _numeric_key(first.start), _numeric_key(last.end)
        for p in pieces[1:]:
            start_key, end_key = _numeric_key(p.start), _numeric_key(p.end)
            if _key_less(start_key, first_key):
                first, first_key = p, start_key
            if _key_less(last_key, end_key):
                last, last_key = p, end_key
        box[v] = (first.start, last.end)
        openness[v] = (first.left_open, last.right_open)
    return box, openness


def _single_argument(f: Expr, symbols: set[Symbol]) -> Optional[tuple[Expr, Expr, Symbol]]:
    """When ``f`` depends on the variables only through one subexpression
    ``u``: ``(F, u, v)`` with ``f = F(v)`` at ``v = u``."""
    v = Symbol('v', real=True)
    candidates = [e for e in preorder_traversal(f) if isinstance(e, Expr) and e is not f
                  and _symbols(e) == symbols and not isinstance(e, Symbol)]

    def operations(e: Expr) -> int:
        count = count_ops(e)
        if not isinstance(count, Integer):
            raise TypeError("count_ops gave %s" % (count,))
        return count.p

    for u in sorted(candidates, key=lambda e: -operations(e)):
        F = replaced(f, {u: v})
        if _symbols(F) == {v}:
            return F, u, v
    return None


def signs_multivariate(f: Expr, symbols: list[Symbol], facts: Facts) -> Optional[Signs]:
    """The signs of an expression of several real variables on the box of
    their assumptions: branch and bound with interval arithmetic,
    monotonicity in every variable (extreme values at the corners), and the
    range of a single inner argument analysed as a univariate problem.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.assumptions.facts import Facts
    >>> from sympy_extras.assumptions.analysis import signs_multivariate
    >>> x, y = Symbol('x'), Symbol('y')
    >>> facts = Facts([x > 0, x < 1, y >= 2])
    >>> signs_multivariate(x - y, [x, y], facts), sorted(signs_multivariate(x*y - 1, [x, y], facts))
    (frozenset({-1}), [-1, 0, 1])
    """
    found = box_of(symbols, facts)
    if found is None:
        return None
    box, openness = found

    def partial_sign(derivative: Expr) -> Optional[Signs]:
        # a partial derivative of one variable is analysed exactly on the
        # (possibly open) domain of that variable; otherwise on the box
        inner = _symbols(derivative)
        if not inner:
            s = certified_sign(derivative)
            return None if s is None else frozenset([s])
        if len(inner) == 1:
            [w] = inner
            domain = domain_of(w, facts)
            return sign_on(derivative, w, domain, 0) if domain is not None else None
        return signs_on_box(derivative, box, weak=True)

    signs = corner_signs(f, box, openness, partial_sign)
    if signs is not None:
        return signs
    signs = signs_on_box(f, box)
    if signs is not None:
        return signs
    single = _single_argument(f, set(symbols))
    if single is not None:
        F, u, v = single
        rng = monotone_range(u, box, openness, partial_sign)
        if rng is None:
            rng = range_on_box(u, box)
        if rng is not None:
            return sign_on(F, v, rng)
    return None


def _real_on(f: Expr, x: Optional[Symbol], domain: Optional[Set], facts: Facts,
             symbols: Optional[list[Symbol]] = None) -> bool:
    """Whether the logarithms and rational powers of ``f`` are real on the
    whole domain: the argument of a logarithm positive, the base of a
    root nonnegative (the bug: ``2 - x**(1/3)`` was found positive on
    ``-1 < x < 1``, where the principal root is not real for ``x < 0``)."""
    if f.has(I):
        # a sign is a property of real values (the bug: exp(-I*x) > 0 was
        # True on -1 < x < 1, from its value 1 at 0 and the absence of zeros)
        return False
    checks: list[tuple[Expr, Signs]] = []
    for node in f.atoms(Log):
        if isinstance(node, Log):
            checks.append((as_expr(node.args[0]), frozenset([1])))
    for node in f.atoms(Pow):
        if isinstance(node, Pow):
            exponent = node.exp
            if isinstance(exponent, Rational) and exponent.q != 1 and node.base.free_symbols:
                checks.append((node.base, frozenset([0, 1]) if exponent.p > 0 else frozenset([1])))
    for inner, allowed in checks:
        if x is not None and domain is not None:
            signs = sign_on(inner, x, domain)
        else:
            signs = signs_multivariate(inner, symbols or [], facts)
        if signs is None or not signs <= allowed:
            return False
    return True


def decide_relational(atom: Relational, facts: Facts) -> Truth:
    """Truth of a relation between expressions of one real variable under
    the facts, by the sign analysis of the difference of its sides.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.functions import sin, exp
    >>> from sympy_extras.assumptions.facts import Facts
    >>> from sympy_extras.assumptions.analysis import decide_relational
    >>> x, y = Symbol('x'), Symbol('y')
    >>> decide_relational(x**2 > 1, Facts([x > 2])), decide_relational(1/(x - 3) < 0, Facts([x > 0, x < 2]))
    (True, True)
    >>> decide_relational(x > 5, Facts([x > 2])) is None
    True
    """
    symbols = _symbols(atom)
    if not symbols:
        return None
    f = atom.lhs - atom.rhs
    signs: Optional[Signs]
    if len(symbols) == 1:
        [x] = symbols
        domain = domain_of(x, facts)
        if domain is None or isinstance(domain, EmptySetClass):
            return None
        if not _real_on(f, x, domain, facts):
            return None
        signs = sign_on(f, x, domain)
    else:
        ordered = sorted(symbols, key=lambda v: v.name)
        if not _real_on(f, None, None, facts, ordered):
            return None
        signs = signs_multivariate(f, ordered, facts)
    if signs is None:
        return None
    kind = type(atom)
    holds: Signs
    if kind is Gt:
        holds = frozenset([1])
    elif kind is Ge:
        holds = frozenset([0, 1])
    elif kind is Lt:
        holds = frozenset([-1])
    elif kind is Le:
        holds = frozenset([-1, 0])
    elif kind is Eq:
        holds = frozenset([0])
    elif kind is Ne:
        holds = frozenset([-1, 1])
    else:
        return None
    if signs <= holds:
        return True
    if not (signs & holds):
        return False
    return None
