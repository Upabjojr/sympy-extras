"""Signs of expressions of one real variable which are not polynomials.

The cylindrical algebraic decomposition decides statements about
polynomials. For an expression `f(x)` with exponentials, logarithms,
trigonometric or other elementary functions, the sign of `f` on the set
where `x` is assumed to lie is found from calculus:

1. `f` must be continuous on the set (SymPy's ``continuous_domain``);
2. the zeros of `f` on the set are computed exactly with ``solveset``;
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

Every answer is a proof modulo the correctness of ``solveset`` and
``continuous_domain``; the undecided cases give ``None``.

The module works on the expressions of the core and imports no SymPy:
:mod:`sympy_extras.assumptions.analysis_sympy` is its face on SymPy
objects. Three steps are not ported and go through SymPy at run time
(the face holds them): ``continuous_domain`` and ``solveset`` for
expressions which are not rational in the variable (the rational ones
are handled by :func:`~sympy_extras.polys.rationalsolve.solveset_rational`),
and the intersection of the solutions of a polynomial assumption which is
not a bound (the native CAD's ``solution_set``) with other intervals,
when the core does not compare their ends.
"""
from __future__ import annotations

from typing import Optional

from sympy_extras.core.accumbounds import AccumBounds, accum_bounds
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import count_ops
from sympy_extras.core.limits import limit
from sympy_extras.core.logic import Boolean, false, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import (Float, I, ImaginaryUnit, Infinity, Integer, NegativeInfinity, Rational,
    number_from_fraction, oo)
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.exponential import Log
from sympy_extras.polys.rationalsolve import solveset_rational
from sympy_extras.sets.sets import (Contains, EmptySetClass, Interval, Reals, Set, Union, intersection, interval,
    union)
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
            # against other numbers): SymPy's intersection
            from .analysis_sympy import intersection_of_core
            found = intersection_of_core(domain, piece)
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


def _continuous_on(f: Expr, x: Symbol, domain: Set) -> bool:
    """Whether ``f`` is continuous on ``domain`` (SymPy's
    ``continuous_domain`` contains it): a rational function natively,
    when no zero of a denominator is in the domain."""
    if _is_rational_in(f, x):
        known = True
        for node in preorder_traversal(f):
            if isinstance(node, Pow) and isinstance(node.exp, Integer) and node.exp.p < 0 \
                    and x in node.base.free_symbols:
                found, poles = _real_zeros_in(node.base, x, domain)
                if not found:
                    known = False
                    break
                if poles:
                    return False
        if known:
            return True
    from .analysis_sympy import continuous_on_of_core
    return continuous_on_of_core(f, x, domain)


def _finite_zeros(f: Expr, x: Symbol, domain: Set) -> Optional[list[Expr]]:
    """The zeros of ``f`` in ``domain`` when ``solveset`` finds a finite
    set of them, ``None`` otherwise."""
    if _is_rational_in(f, x):
        found, zeros = _real_zeros_in(f, x, domain)
        if found:
            return zeros
    from .analysis_sympy import finite_zeros_of_core
    return finite_zeros_of_core(f, x, domain)


def _has_discrete_zeros(f: Expr, x: Symbol, piece: Interval) -> bool:
    """Whether the zeros of ``f`` on the piece are isolated points (a
    finite set, or a periodic family)."""
    if _is_rational_in(f, x) and _real_zeros_in(f, x, piece)[0]:
        return True
    from .analysis_sympy import has_discrete_zeros_of_core
    return has_discrete_zeros_of_core(f, x, piece)


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
