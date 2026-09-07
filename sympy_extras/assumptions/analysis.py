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
   certified numerical evaluation (the value is computed with a margin
   over its error bound before its sign is trusted);
3. when the zeros cannot be found, interval arithmetic (SymPy's
   ``AccumBounds``) bounds `f` on the set, the critical points of `f'`
   split the set into monotone pieces whose signs follow from the limits
   at the ends, and finally the zeros are isolated by bisection with
   rigorous interval arithmetic (:mod:`sympy_extras.assumptions.intervals`).

Expressions of several variables are handled on the box of their
assumptions by branch and bound with interval arithmetic, by monotonicity
in every variable, or through the range of a single inner argument.

Every answer is a proof modulo the correctness of ``solveset`` and
``continuous_domain``; the undecided cases give ``None``.
"""
from __future__ import annotations

from typing import Optional

from sympy.calculus.util import continuous_domain
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.expr import Expr
from sympy.core.function import count_ops
from sympy.core.traversal import preorder_traversal
from sympy.core.numbers import Rational, Float
from sympy.core.relational import Relational, Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Reals, Integers, Naturals, Naturals0, ImageSet
from sympy.sets.sets import Set, Interval, FiniteSet, Union as SetUnion, Intersection, EmptySet
from sympy.series.limits import limit
from sympy.solvers.solveset import solveset

from sympy_extras._timeout import attempt
from sympy_extras._typing import Sign, Truth, as_expr, as_set, free_symbols
from sympy_extras.polys.cad import solution_set
from sympy_extras.settings import settings

from .facts import Facts, to_polynomial
from .intervals import Box, Openness, signs_on_box, corner_signs, range_on_box, monotone_range, isolate_signs

__all__ = ['certified_sign', 'domain_of', 'sign_on', 'signs_multivariate', 'box_of', 'decide_relational']

#: the possible signs of an expression on a set
Signs = frozenset[Sign]


def certified_sign(value: Expr) -> Optional[Sign]:
    """The sign of a constant expression: from SymPy's exact knowledge, or
    from numerical evaluation with a margin over the error bound when the
    numerical checks are enabled; ``None`` when undecided."""
    if free_symbols(value):
        return None
    if value is S.Infinity:
        return 1
    if value is S.NegativeInfinity:
        return -1
    if value.is_zero:
        return 0
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    if not settings.numerical_checks or value.is_real is False:
        return None
    for digits in (settings.precision, 2*settings.precision, 4*settings.precision):
        approximation = value.evalf(digits)
        if not isinstance(approximation, Float) or not approximation.is_finite:
            return None
        # SymPy tracks the precision of the result: a value which is not
        # zero to that precision has a certain sign
        if abs(approximation) > Float(10)**(-(digits//2)):
            return 1 if approximation > 0 else -1
    return None


def certified_compare(a: Expr, b: Expr) -> Optional[Sign]:
    """The sign of ``a - b`` for constants, infinities included."""
    if a is S.Infinity or b is S.NegativeInfinity:
        return 0 if a == b else 1
    if a is S.NegativeInfinity or b is S.Infinity:
        return 0 if a == b else -1
    return certified_sign(as_expr(a - b))


def _interval_of_bound(atom: Boolean, x: Symbol) -> Optional[Set]:
    """The interval described by a bound ``x rel c`` with a constant
    ``c``."""
    if not isinstance(atom, Relational) or free_symbols(atom) != {x}:
        return None
    lhs, rhs = as_expr(atom.lhs), as_expr(atom.rhs)
    if lhs == x and not free_symbols(rhs):
        c, flipped = rhs, False
    elif rhs == x and not free_symbols(lhs):
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
        return Interval.open(c, S.Infinity)
    if kind is Ge:
        return Interval(c, S.Infinity)
    if kind is Lt:
        return Interval.open(S.NegativeInfinity, c)
    if kind is Le:
        return Interval(S.NegativeInfinity, c)
    if kind is Eq:
        return FiniteSet(c)
    if kind is Ne:
        return SetUnion(Interval.open(S.NegativeInfinity, c), Interval.open(c, S.Infinity))
    return None


def domain_of(x: Symbol, facts: Facts) -> Optional[Set]:
    """The set of real values of ``x`` allowed by the facts which only
    concern ``x``, as a union of intervals; ``None`` when some fact about
    ``x`` involves other symbols or cannot be turned into intervals."""
    if x not in facts.real:
        return None
    domain: Set = S.Reals
    for atom in facts.conjuncts:
        symbols = free_symbols(atom)
        if x not in symbols:
            continue
        if symbols != {x}:
            return None
        piece: Optional[Set] = None
        if isinstance(atom, Contains):
            candidate = as_set(atom.args[1])
            if isinstance(candidate, (Interval, Reals)):
                piece = candidate
        if piece is None:
            piece = _interval_of_bound(atom, x)
        if piece is None and to_polynomial(atom, facts.real) is not None:
            piece = attempt(lambda: solution_set(atom, x), settings.timeout)
        if piece is None:
            return None
        domain = Intersection(domain, piece)
    domain = as_set(domain)
    if isinstance(domain, EmptySet):
        return domain
    if isinstance(domain, (Interval, Reals)):
        return domain
    if isinstance(domain, SetUnion) and all(isinstance(a, Interval) for a in domain.args):
        return domain
    return None


def _pieces(domain: Set) -> Optional[list[Interval]]:
    if isinstance(domain, Reals):
        return [Interval(S.NegativeInfinity, S.Infinity)]
    if isinstance(domain, Interval):
        return [domain]
    if isinstance(domain, SetUnion):
        pieces: list[Interval] = []
        for a in domain.args:
            if not isinstance(a, Interval):
                return None
            pieces.append(a)
        return pieces
    return None


def _between(p: Expr, q: Expr) -> Optional[Expr]:
    """A rational number strictly between the constants ``p < q``."""
    if p is S.NegativeInfinity and q is S.Infinity:
        return S.Zero
    for digits in (settings.precision, 2*settings.precision):
        if p is S.NegativeInfinity:
            candidate = as_expr(Rational(q.evalf(digits)) - 1)
        elif q is S.Infinity:
            candidate = as_expr(Rational(p.evalf(digits)) + 1)
        else:
            candidate = as_expr(Rational((p.evalf(digits) + q.evalf(digits))/2))
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


def sign_on(f: Expr, x: Symbol, domain: Set, depth: int = 1) -> Optional[Signs]:
    """The set of signs which ``f`` takes on ``domain`` (a union of
    intervals), or ``None`` when it cannot be established."""
    pieces = _pieces(domain)
    if pieces is None:
        return None
    if not f.has(x):
        s = certified_sign(f)
        return None if s is None else frozenset([s])
    continuity = attempt(lambda: as_set(continuous_domain(f, x, S.Reals)), settings.timeout)
    if continuity is None or domain.is_subset(continuity) is not True:
        return None
    zeros = attempt(lambda: as_set(solveset(f, x, domain)), settings.timeout)
    if isinstance(zeros, (FiniteSet, EmptySet)):
        zero_list = [as_expr(z) for z in zeros.args] if isinstance(zeros, FiniteSet) else []
        if all(z.is_real for z in zero_list):
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
    """The signs allowed by interval arithmetic (SymPy's ``AccumBounds``)
    on every piece: an over-approximation, so only a one-sided answer is
    informative."""
    signs: set[Sign] = set()
    for piece in pieces:
        bounds = attempt(lambda: f.subs(x, AccumBounds(piece.start, piece.end)), settings.timeout)
        if bounds is None:
            return None
        if isinstance(bounds, AccumBounds):
            low, high = as_expr(bounds.min), as_expr(bounds.max)
        elif isinstance(bounds, Expr) and not free_symbols(bounds):
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
        a, b = as_expr(piece.start), as_expr(piece.end)
        inside = [z for z in zeros if certified_compare(z, a) == 1 and certified_compare(b, z) == 1]
        ordered = _sorted_constants([a] + inside + [b])
        if ordered is None:
            return None
        if len(ordered) < 2:
            # a single point: its sign is the value there
            s = certified_sign(as_expr(f.subs(x, ordered[0])))
            if s is None:
                return None
            signs.add(s)
            continue
        for p, q in zip(ordered, ordered[1:]):
            sample = _between(p, q)
            if sample is None:
                return None
            s = certified_sign(as_expr(f.subs(x, sample)))
            if s is None:
                return None
            signs.add(s)
    return frozenset(signs)


def _endpoint_value(f: Expr, x: Symbol, point: Expr, direction: str) -> Optional[Expr]:
    value = attempt(lambda: as_expr(limit(f, x, point, direction)), settings.timeout)
    if value is None or free_symbols(value):
        return None
    return value


def _is_discrete(zeros: Set) -> bool:
    if isinstance(zeros, (FiniteSet, EmptySet)):
        return True
    if isinstance(zeros, ImageSet):
        return isinstance(zeros.base_set, (Integers, Naturals, Naturals0))
    if isinstance(zeros, SetUnion):
        return all(_is_discrete(as_set(a)) for a in zeros.args)
    if isinstance(zeros, Intersection):
        return any(_is_discrete(as_set(a)) for a in zeros.args)
    return False


def _has_discrete_zeros(f: Expr, x: Symbol, piece: Interval) -> bool:
    zeros = attempt(lambda: as_set(solveset(f, x, piece)), settings.timeout)
    return zeros is not None and _is_discrete(zeros)


def _value_at(f: Expr, x: Symbol, point: Expr, direction: str) -> Optional[Expr]:
    """The value of a continuous ``f`` at a finite point, or its limit at
    an infinite one."""
    if point.is_finite:
        value = as_expr(f.subs(x, point))
        if not free_symbols(value) and value.is_finite is not False:
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
    derivative = as_expr(f.diff(x))
    signs: set[Sign] = set()
    for piece in pieces:
        a, b = as_expr(piece.start), as_expr(piece.end)
        critical = attempt(lambda: as_set(solveset(derivative, x, piece)), settings.timeout)
        if isinstance(critical, (FiniteSet, EmptySet)):
            points = [as_expr(c) for c in critical.args] if isinstance(critical, FiniteSet) else []
            if not all(c.is_real for c in points):
                return None
            inside = [c for c in points if certified_compare(c, a) == 1 and certified_compare(b, c) == 1]
            ordered = _sorted_constants([a] + inside + [b])
            if ordered is None:
                return None
            for p, q in zip(ordered, ordered[1:]):
                sample = _between(p, q)
                if sample is None:
                    return None
                slope = certified_sign(as_expr(derivative.subs(x, sample)))
                if slope is None or slope == 0:
                    return None
                part = _monotone_signs(f, x, p, q, slope == 1)
                if part is None:
                    return None
                signs.update(part)
            for c in inside:
                value = certified_sign(as_expr(f.subs(x, c)))
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


def box_of(symbols: list[Symbol], facts: Facts) -> Optional[tuple[Box, Openness]]:
    """The box of the values of the symbols allowed by the facts which
    concern one symbol at a time (the hull of the pieces), and for every
    symbol whether the range is open at each end; ``None`` when a symbol
    is not known to be real."""
    box: Box = {}
    openness: Openness = {}
    for v in symbols:
        domain = domain_of(v, facts)
        if domain is None:
            return None
        pieces = _pieces(domain)
        if not pieces:
            return None
        first = min(pieces, key=lambda p: as_expr(p.start).evalf(20))
        last = max(pieces, key=lambda p: as_expr(p.end).evalf(20))
        box[v] = (as_expr(first.start), as_expr(last.end))
        openness[v] = (bool(first.left_open), bool(last.right_open))
    return box, openness


def _single_argument(f: Expr, symbols: set[Symbol]) -> Optional[tuple[Expr, Expr, Symbol]]:
    """When ``f`` depends on the variables only through one subexpression
    ``u``: ``(F, u, v)`` with ``f = F(v)`` at ``v = u``."""
    v = Symbol('v', real=True)
    candidates = [e for e in preorder_traversal(f) if isinstance(e, Expr) and e is not f
                  and free_symbols(e) == symbols and not isinstance(e, Symbol)]
    for u in sorted(candidates, key=lambda e: -int(count_ops(e))):
        F = as_expr(f.xreplace({u: v}))
        if free_symbols(F) == {v}:
            return F, as_expr(u), v
    return None


def signs_multivariate(f: Expr, symbols: list[Symbol], facts: Facts) -> Optional[Signs]:
    """The signs of an expression of several real variables on the box of
    their assumptions: branch and bound with interval arithmetic,
    monotonicity in every variable (extreme values at the corners), and the
    range of a single inner argument analysed as a univariate problem."""
    found = box_of(symbols, facts)
    if found is None:
        return None
    box, openness = found

    def partial_sign(derivative: Expr) -> Optional[Signs]:
        # a partial derivative of one variable is analysed exactly on the
        # (possibly open) domain of that variable; otherwise on the box
        inner = free_symbols(derivative)
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


def decide_relational(atom: Relational, facts: Facts) -> Truth:
    """Truth of a relation between expressions of one real variable under
    the facts, by the sign analysis of the difference of its sides.

    Examples
    ========

    >>> from sympy import sin, exp, pi, Abs
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions import ask, refine
    >>> ask(sin(x) > 0, (x > 0) & (x < pi))
    True
    >>> ask(exp(x) > 2, x > 1)
    True
    >>> refine(Abs(sin(x)), (x > 0) & (x < pi))
    sin(x)
    >>> ask(x*exp(x) > 1, x > 1)
    True
    >>> from sympy.abc import y
    >>> ask(exp(x) + y > 1, (x > 0) & (y > 0))
    True
    >>> ask(sin(x*y) > 0, (x > 0) & (x < 1) & (y > 0) & (y < 3))
    True
    """
    symbols = free_symbols(atom)
    if not symbols:
        return None
    f = as_expr(atom.lhs - atom.rhs)
    signs: Optional[Signs]
    if len(symbols) == 1:
        [x] = symbols
        domain = domain_of(x, facts)
        if domain is None or isinstance(domain, EmptySet):
            return None
        signs = sign_on(f, x, domain)
    else:
        signs = signs_multivariate(f, sorted(symbols, key=lambda v: v.name), facts)
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
