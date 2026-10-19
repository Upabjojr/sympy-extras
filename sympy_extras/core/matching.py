"""Pattern matching: the algorithm behind ``Basic.match``.

A pattern is a tree containing :class:`~sympy_extras.core.symbol.Wild`
symbols (and :class:`~sympy_extras.core.function.WildFunction` heads);
matching an expression against it finds bindings of the wilds for which
the pattern equals the expression. The search is a backtracking
generator: a wild-free pattern must be equal to the expression; a wild
binds to the expression when its ``exclude`` and ``properties`` allow it;
a node is matched argument by argument, except sums and products, which
are matched as multisets, the terms or factors of the pattern which
contain wilds being tried against those of the expression and the lone
wilds of the pattern absorbing what is left (the identity ``0`` or ``1``
when nothing is left, unless ``exact``). A power matches a non-power as a
first power, and a sum or product pattern with a single lone wild matches
by solving (``x**3`` against ``a*x**2`` gives ``a = x``), as SymPy does.
Structural matches are searched before solved ones, so the natural
binding is found first.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Wild and match*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.match
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

from .basic import Basic, MatchDict
from .expr import Expr, factors_of, terms_of

__all__ = ['match']


def match(expr: Basic, pattern: Basic, bindings: Optional[MatchDict] = None, exact: bool = False) -> Optional[MatchDict]:
    """The bindings of the wilds of ``pattern`` for which it equals
    ``expr`` (extending ``bindings``), ``None`` when there are none.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Wild, Function
    >>> from sympy_extras.core.matching import match
    >>> x, y = symbols('x y')
    >>> a, b = Wild('a'), Wild('b')
    >>> match(x*y + 1, a*x + b) == {a: y, b: 1}
    True
    >>> match(x**3, a*x**2) == {a: x}
    True
    >>> f = Function('f')
    >>> match(f(x, y), f(a, b)) == {a: x, b: y}, match(f(x), f(a, b))
    (True, None)
    """
    start: MatchDict = dict(bindings or {})
    for solve in (False, True):
        for found in _matches(expr, pattern, start, exact, solve):
            return found
    return None


def _has_wild(pattern: Basic) -> bool:
    from .function import WildFunction
    from .symbol import Wild
    return pattern.has(Wild, WildFunction)


def _matches(expr: Basic, pattern: Basic, bindings: MatchDict, exact: bool, solve: bool) -> Iterator[MatchDict]:
    from .add import Add
    from .function import Application, WildFunction
    from .mul import Mul
    from .power import Pow
    from .symbol import Wild
    if isinstance(pattern, Wild):
        if pattern in bindings:
            if bindings[pattern] == expr:
                yield bindings
        elif pattern.accepts(expr):
            yield {**bindings, pattern: expr}
        return
    if isinstance(pattern, WildFunction):
        if isinstance(expr, Application) and pattern.accepts(len(expr.args)):
            if pattern in bindings:
                if bindings[pattern] == expr:
                    yield bindings
            else:
                yield {**bindings, pattern: expr}
        return
    if not _has_wild(pattern):
        if expr == pattern:
            yield bindings
        return
    if isinstance(pattern, Add) and isinstance(expr, Expr):
        yield from _match_assoc(expr, pattern, bindings, exact, solve, True)
        return
    if isinstance(pattern, Mul) and isinstance(expr, Expr):
        yield from _match_assoc(expr, pattern, bindings, exact, solve, False)
        return
    if isinstance(pattern, Pow) and isinstance(expr, Expr):
        yield from _match_pow(expr, pattern, bindings, exact, solve)
        return
    if expr.func != pattern.func or len(expr.args) != len(pattern.args):
        return
    yield from _match_sequence(expr.args, pattern.args, bindings, exact, solve)


def _match_sequence(exprs: Sequence[Basic], patterns: Sequence[Basic], bindings: MatchDict,
                    exact: bool, solve: bool) -> Iterator[MatchDict]:
    if not patterns:
        yield bindings
        return
    for found in _matches(exprs[0], patterns[0], bindings, exact, solve):
        yield from _match_sequence(exprs[1:], patterns[1:], found, exact, solve)


def _combine(parts: list[Expr], is_add: bool) -> Expr:
    from .add import add
    from .mul import mul
    return add(*parts) if is_add else mul(*parts)


def _match_assoc(expr: Expr, pattern: Expr, bindings: MatchDict, exact: bool, solve: bool,
                 is_add: bool) -> Iterator[MatchDict]:
    """Matching of a sum or product pattern against ``expr``."""
    from .numbers import Integer
    from .sorting import default_sort_key
    from .symbol import Wild
    subject = list(terms_of(expr) if is_add else factors_of(expr))
    identity: Expr = Integer(0) if is_add else Integer(1)
    lone: list[Wild] = []
    others: list[Basic] = []
    for p in pattern.args:
        if isinstance(p, Wild):
            if p in bindings:
                others.append(bindings[p])
            else:
                lone.append(p)
        else:
            others.append(p)
    found_any = False
    for found, remaining in _match_multiset(subject, others, bindings, exact, solve, identity):
        unbound = [w for w in lone if w not in found]
        if not unbound:
            if not remaining:
                found_any = True
                yield found
            continue
        if exact and not remaining:
            continue
        remaining = sorted(remaining, key=default_sort_key)
        assignments: list[tuple[Wild, Expr]] = []
        tail = unbound[1:]
        rest = list(remaining)
        for w in reversed(tail):
            value = rest.pop() if rest else identity
            assignments.append((w, value))
        assignments.append((unbound[0], _combine(rest, is_add)))
        result = dict(found)
        accepted = True
        for w, value in assignments:
            if w in result and result[w] != value:
                accepted = False
                break
            if not w.accepts(value) or (exact and value == identity):
                accepted = False
                break
            result[w] = value
        if accepted:
            found_any = True
            yield result
    if found_any or not solve:
        return
    # solving: w + rest = expr or w*rest = expr with a single lone wild and
    # wild-free other terms, as SymPy does when the subject is not of the
    # other kind (a product for a sum pattern) and mentions every symbol
    # of the rest (``x**3`` against ``a*x**2`` gives ``x``, ``y`` gives nothing)
    unbound = [w for w in lone if w not in bindings]
    if len(unbound) != 1 or any(_has_wild(o) for o in others):
        return
    if (is_add and expr.is_Mul) or (not is_add and expr.is_Add):
        return
    rest_value = _combine([_as_expr(o) for o in others], is_add)
    if not (expr.is_number or rest_value.free_symbols <= expr.free_symbols):
        return
    if not is_add and expr.is_Mul and not _bases_present(expr, rest_value):
        return
    from .add import sub
    from .mul import div
    value = sub(expr, rest_value) if is_add else div(expr, rest_value)
    if unbound[0].accepts(value) and not (exact and value == identity):
        yield {**bindings, unbound[0]: value}


def _bases_present(expr: Expr, rest: Expr) -> bool:
    """Whether every factor ``b**e`` of ``rest`` has a factor of ``expr``
    with the base ``b`` and an exponent of the sign of ``e`` (SymPy solves
    ``2*x**3`` against ``a*x**2`` but not ``2/x`` against ``a*x``)."""
    from .numbers import Number, Rational
    subject = {}
    for f in factors_of(expr):
        b, e = f.as_base_exp()
        subject[b] = e
    for f in factors_of(rest):
        if isinstance(f, Number):
            continue
        b, e = f.as_base_exp()
        if b not in subject:
            return False
        present = subject[b]
        if isinstance(e, Rational) and isinstance(present, Rational) and (e.p > 0) != (present.p > 0):
            return False
    return True


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def _match_multiset(subject: list[Expr], patterns: list[Basic], bindings: MatchDict, exact: bool,
                    solve: bool, identity: Expr) -> Iterator[tuple[MatchDict, list[Expr]]]:
    """The ways of matching each pattern term against a distinct subject
    term (or against the identity, for a term with wilds), with the
    bindings and the subject terms left over."""
    if not patterns:
        yield (bindings, subject)
        return
    p, rest = patterns[0], patterns[1:]
    # the subject terms of the class of the pattern term are tried first
    order = sorted(range(len(subject)), key=lambda i: type(subject[i]) is not type(p))
    for i in order:
        for found in _matches(subject[i], p, bindings, exact, solve):
            yield from _match_multiset(subject[:i] + subject[i + 1:], rest, found, exact, solve, identity)
    if _has_wild(p) and not exact:
        for found in _matches(identity, p, bindings, exact, solve):
            yield from _match_multiset(subject, rest, found, exact, solve, identity)


def _match_pow(expr: Expr, pattern: Expr, bindings: MatchDict, exact: bool, solve: bool) -> Iterator[MatchDict]:
    from .numbers import Integer, Rational, number_from_fraction
    from .power import Pow, pow
    if not isinstance(pattern, Pow):
        return
    if isinstance(expr, Pow):
        yield from _match_sequence(expr.args, pattern.args, bindings, exact, solve)
        if not solve:
            return
        # x**4 against a**2: a = x**2
        if isinstance(expr.exp, Rational) and isinstance(pattern.exp, Rational) and not _has_wild(pattern.exp):
            ratio = expr.exp.as_fraction() / pattern.exp.as_fraction()
            if ratio.denominator == 1 and ratio != 1:
                yield from _matches(pow(expr.base, number_from_fraction(ratio)), pattern.base, bindings, exact, solve)
        return
    base, exponent = expr.as_base_exp()
    if not (exponent.is_Number and exponent == 1):
        # a product with one exponent, x**2*y**2 as (x*y)**2
        yield from _match_sequence((base, exponent), pattern.args, bindings, exact, solve)
    yield from _match_sequence((expr, Integer(1)), pattern.args, bindings, exact, solve)
    if expr.is_Number and expr == 1 and not _has_wild(pattern.base):
        # the identity of a product as the zeroth power of the base
        yield from _matches(Integer(0), pattern.exp, bindings, exact, solve)
    if solve and isinstance(pattern.exp, Rational) and not _has_wild(pattern.exp) and pattern.exp.p != 0:
        inverse = Rational(pattern.exp.q, pattern.exp.p)
        yield from _matches(pow(expr, inverse), pattern.base, bindings, exact, solve)
