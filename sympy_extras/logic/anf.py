"""The algebraic normal form of Boolean formulas, on the objects of the
core: a typed port of ``to_anf``, ``is_anf``, ``distribute_xor_over_and``,
``ANFform``, ``anf_coeffs``, ``bool_monomial`` and of the evaluation of
``Xor`` of ``sympy.logic.boolalg``.

A formula is in algebraic normal form (its Zhegalkin polynomial [1]_)
when it is ``true``, ``false``, a conjunction of variables or an
exclusive disjunction of such conjunctions and ``true``; with ``And`` as
the product and ``Xor`` as the sum modulo 2, every formula has exactly
one such form. :func:`to_anf` rewrites a formula into it as SymPy does:
``Or`` becomes the exclusive disjunction of the conjunctions of the
non-empty subsets of its arguments, ``Not(a)`` is ``a ^ True``,
``Implies(a, b)`` is ``True ^ a ^ (a & b)``, ``Equivalent`` the
exclusive disjunction of the conjunction of its arguments and that of
their negations, and the conjunctions are distributed over the exclusive
disjunctions (:func:`distribute_xor_over_and`). :func:`xor_evaluated` is
SymPy's ``Xor`` constructor: equal arguments cancel in pairs, ``false``
is dropped, two relations with the same canonical form cancel and two
complementary ones give ``true``; with ``remove_true`` (the default) a
``true`` argument is taken out as a negation, without it it stays an
argument (``a ^ True``), as the normal form needs.

:func:`anf_coeffs` computes the coefficients of the Zhegalkin polynomial
from the truth table by the binary Moebius transform, and
:func:`ANFform` builds the polynomial from them.

A relation, or any Boolean which is not a connective (``Contains``), is a
variable of the normal form: SymPy calls a ``to_anf`` method which these
classes do not have when a side of a relation is not an atom
(``to_anf(Xor(a, x + y > 1))`` raises ``AttributeError`` there).

Examples
========

>>> from sympy_extras.core import symbols, and_, or_, not_
>>> from sympy_extras.logic.anf import to_anf, ANFform
>>> a, b, c = symbols('a b c')
>>> to_anf(not_(a)), to_anf(and_(or_(a, b), not_(c)))
(a ^ True, a ^ b ^ (a & b) ^ (a & c) ^ (b & c) ^ (a & b & c))
>>> ANFform([a, b], [0, 1, 1, 1])
a ^ b ^ (a & b)

References
==========

.. [1] https://en.wikipedia.org/wiki/Zhegalkin_polynomial
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/logic/boolalg.py (Xor.__new__, to_anf, is_anf,
# BooleanFunction._to_anf, And.to_anf, Or.to_anf, Not.to_anf, Implies.to_anf,
# Equivalent.to_anf, distribute_xor_over_and, _distribute, ANFform,
# anf_coeffs, bool_monomial, _convert_to_varsANF, And._new_args_filter)
# (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing on the core's Booleans; relations and the other
# non-connectives are variables of the normal form (SymPy fails on a
# relation whose sides are not atoms); a deterministic order of the
# pairs of relations of Xor.
# ---------------------------------------------------------------------
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product
from typing import Sequence, Union

from sympy_extras.core.logic import (And, Boolean, BooleanAtom, BooleanFunction, BooleanLike, Equivalent, ITE,
                                     Implies, Not, Or, Xor, and_, false, ite, not_, to_boolean, true)
from sympy_extras.core.relational import Relational
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Symbol
from sympy_extras.logic.boolalg import is_literal

__all__ = ['xor_evaluated', 'and_evaluated', 'is_anf', 'to_anf', 'distribute_xor_over_and', 'anf_coeffs',
           'bool_monomial', 'ANFform']


def _ordered_booleans(items: Sequence[Boolean]) -> list[Boolean]:
    return [to_boolean(a) for a in ordered(items)]


def _toggle(found: list[Boolean], b: Boolean) -> None:
    """``b`` added to the arguments, or removed when it is there."""
    if b in found:
        found.remove(b)
    else:
        found.append(b)


@lru_cache(maxsize=4096)
def _canonical_forms(r: Relational) -> tuple[Relational, Relational]:
    """The canonical forms of a relation and of its negation (remembered:
    the distribution over ``Xor`` builds the same conjunctions many
    times)."""
    return r.canonical, r.negated.canonical


# PORTED FROM SYMPY: sympy/logic/boolalg.py:Xor.__new__
def xor_evaluated(*args: BooleanLike, remove_true: bool = True) -> Boolean:
    """SymPy's ``Xor(*args, remove_true=remove_true)``: the arguments of
    nested ``Xor`` taken in, ``false`` dropped, equal arguments and pairs
    of relations with the same canonical form cancelled, a pair of
    complementary relations replaced by ``true``; a ``true`` argument
    negates the rest when ``remove_true``, and stays an argument
    otherwise.

    Examples
    ========

    >>> from sympy_extras.core import symbols, true
    >>> from sympy_extras.logic.anf import xor_evaluated
    >>> a, b, x = symbols('a b x')
    >>> xor_evaluated(a, b, true), xor_evaluated(a, b, true, remove_true=False)
    (~(a ^ b), a ^ b ^ True)
    >>> xor_evaluated(x > 1, 1 < x, a), xor_evaluated(x > 1, x <= 1, a)
    (a, ~a)
    """
    found: list[Boolean] = []
    for a in args:
        b = to_boolean(a)
        if isinstance(b, BooleanAtom):
            if not b:
                continue
            b = true
        if isinstance(b, Xor):
            for c in b.args:
                _toggle(found, c)
        else:
            _toggle(found, b)
    # the relations: (relation, canonical form, canonical form of the negation)
    relations = [(r,) + _canonical_forms(r) for r in _ordered_booleans(found) if isinstance(r, Relational)]
    odd = False
    used: set[int] = set()
    remove: list[tuple[Relational, Relational]] = []
    for i, (r, c, nc) in enumerate(relations):
        if i in used:
            continue
        for j in range(i + 1, len(relations)):
            if j in used:
                continue
            rj, cj, _ = relations[j]
            if cj == nc:
                odd = not odd
            elif cj != c:
                continue
            used.update((i, j))
            remove.append((r, rj))
            break
    if odd:
        _toggle(found, true)
    for r1, r2 in remove:
        found.remove(r1)
        found.remove(r2)
    if not found:
        return false
    if len(found) == 1:
        return found[0]
    if true in found and remove_true:
        found.remove(true)
        return not_(xor_evaluated(*found))
    return Xor(*_ordered_booleans(found))


# PORTED FROM SYMPY: sympy/logic/boolalg.py:And._new_args_filter
def and_evaluated(*args: BooleanLike) -> Boolean:
    """SymPy's ``And(*args)``: the core's :func:`~sympy_extras.core.logic.and_`
    with the relations compared by their canonical forms (``(x > 1) &
    (1 < x)`` is ``x > 1``, ``(x > 1) & (1 >= x)`` is ``false``).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.anf import and_evaluated
    >>> x, a = symbols('x a')
    >>> and_evaluated(x > 1, 1 < x, a), and_evaluated(x > 1, 1 >= x)
    (a & (x > 1), False)
    """
    r = and_(*args)
    if not isinstance(r, And) or sum(1 for a in r.args if isinstance(a, Relational)) < 2:
        return r
    kept: list[Boolean] = []
    canonical: list[Relational] = []
    for a in r.args:   # (sorted by and_)
        if isinstance(a, Relational):
            c, nc = _canonical_forms(a)
            if c in canonical:
                continue
            if nc in canonical:
                return false
            canonical.append(c)
        kept.append(a)
    return and_(*kept) if len(kept) < len(r.args) else r


def _is_variable(expr: Boolean) -> bool:
    """A variable of the normal form: anything which is not a connective
    (SymPy's ``is_literal`` without its negation)."""
    return not isinstance(expr, BooleanFunction)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:is_anf
def is_anf(expr: Boolean) -> bool:
    """Whether ``expr`` is in algebraic normal form: ``true``, ``false``, a
    variable, a conjunction of symbols or an exclusive disjunction of
    those and ``true``, without negations.

    Examples
    ========

    >>> from sympy_extras.core import symbols, true, Not, Xor, And
    >>> from sympy_extras.logic.anf import is_anf
    >>> a, b, c = symbols('a b c')
    >>> is_anf(true), is_anf(a), is_anf(And(a, b, c)), is_anf(Xor(a, Not(b)))
    (True, True, True, False)
    """
    if is_literal(expr) and not isinstance(expr, Not):
        return True
    if isinstance(expr, And):
        return all(isinstance(a, Symbol) for a in expr.args)
    if isinstance(expr, Xor):
        for a in expr.args:
            if isinstance(a, And):
                if not all(isinstance(s, Symbol) for s in a.args):
                    return False
            elif is_literal(a):
                if isinstance(a, Not):
                    return False
            else:
                return False
        return True
    return False


def _anf_arguments(args: Sequence[Boolean], deep: bool) -> list[Boolean]:
    """The arguments of ``BooleanFunction._to_anf``: in normal form when
    ``deep`` and they are connectives which are not literals, or
    negations."""
    found: list[Boolean] = []
    for a in args:
        if deep and isinstance(a, BooleanFunction) and (not is_literal(a) or isinstance(a, Not)):
            a = _method_anf(a, deep)
        found.append(a)
    return found


def _xor_anf(args: Sequence[Boolean], deep: bool) -> Boolean:
    """SymPy's ``Xor._to_anf``."""
    return xor_evaluated(*_anf_arguments(args, deep), remove_true=False)


def _method_anf(expr: Boolean, deep: bool) -> Boolean:
    """The ``to_anf`` method of the class of ``expr``."""
    if isinstance(expr, And):
        if not deep:
            return expr
        return distribute_xor_over_and(and_evaluated(*_anf_arguments(expr.args, deep)))
    if isinstance(expr, Or):
        subsets = (and_evaluated(*subset) for k in range(1, len(expr.args) + 1)
                   for subset in combinations(expr.args, k))
        return xor_evaluated(*[to_anf(s, deep) if deep else s for s in subsets], remove_true=False)
    if isinstance(expr, Not):
        return _xor_anf([true, expr.arg], deep)
    if isinstance(expr, Implies):
        a, b = expr.args
        return _xor_anf([true, a, and_evaluated(a, b)], deep)
    if isinstance(expr, Equivalent):
        conjunction = and_evaluated(*expr.args)
        negations = distribute_xor_over_and(and_evaluated(*[to_anf(not_(a), False) for a in expr.args]))
        return _xor_anf([conjunction, negations], deep)
    if isinstance(expr, Xor):
        return _xor_anf(expr.args, deep)
    if isinstance(expr, ITE):
        c, a, b = _anf_arguments(expr.args, deep)
        return ite(c, a, b)
    return expr


# PORTED FROM SYMPY: sympy/logic/boolalg.py:to_anf
def to_anf(expr: Boolean, deep: bool = True) -> Boolean:
    """The algebraic normal form of ``expr`` (see the module
    documentation); with ``deep`` false, the arguments of the formula are
    taken as variables.

    Examples
    ========

    >>> from sympy_extras.core import symbols, implies, equivalent, not_
    >>> from sympy_extras.logic.anf import to_anf
    >>> a, b, c = symbols('a b c')
    >>> to_anf(implies(not_(a), equivalent(b, c)), deep=False)
    True ^ ~a ^ (~a & (Equivalent(b, c)))
    >>> to_anf(implies(a, b)), to_anf(equivalent(a, b))
    (a ^ True ^ (a & b), a ^ b ^ True)
    """
    if is_anf(expr) or _is_variable(expr):
        return expr
    return _method_anf(expr, deep)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:distribute_xor_over_and
def distribute_xor_over_and(expr: Boolean) -> Boolean:
    """The conjunctions distributed over the exclusive disjunctions of
    ``expr`` (a formula of conjunctions and exclusive disjunctions of
    literals): an equivalent exclusive disjunction, not simplified.

    Examples
    ========

    >>> from sympy_extras.core import symbols, And, Xor, not_
    >>> from sympy_extras.logic.anf import distribute_xor_over_and
    >>> a, b, c = symbols('a b c')
    >>> distribute_xor_over_and(And(Xor(not_(a), b), c))
    (b & c) ^ (c & ~a)
    """
    if isinstance(expr, And):
        for arg in expr.args:
            if isinstance(arg, Xor):
                rest = and_evaluated(*[a for a in expr.args if a is not arg])
                return xor_evaluated(*[distribute_xor_over_and(and_evaluated(c, rest)) for c in arg.args],
                                     remove_true=False)
        return expr
    if isinstance(expr, Xor):
        return xor_evaluated(*[distribute_xor_over_and(a) for a in expr.args], remove_true=False)
    return expr


# PORTED FROM SYMPY: sympy/logic/boolalg.py:anf_coeffs
def anf_coeffs(truthvalues: Sequence[int]) -> list[int]:
    """The coefficients (0 or 1) of the Zhegalkin polynomial whose truth
    table is ``truthvalues`` (of length a power of two): the coefficient
    of index ``k`` is that of the conjunction of the variables of the
    binary digits of ``k`` (the first variable the most significant).

    Examples
    ========

    >>> from sympy_extras.logic.anf import anf_coeffs
    >>> anf_coeffs([0, 1, 1, 0, 0, 1, 0, 1])
    [0, 1, 1, 0, 0, 0, 1, 0]
    """
    n = len(truthvalues).bit_length() - 1
    if len(truthvalues) != 2**n:
        raise ValueError("The number of truth values must be a power of two, got %d" % len(truthvalues))
    coeffs: list[list[int]] = [[v] for v in truthvalues]
    for i in range(n):
        tmp: list[list[int]] = []
        for j in range(2**(n - i - 1)):
            tmp.append(coeffs[2*j] + [p ^ q for p, q in zip(coeffs[2*j], coeffs[2*j + 1])])
        coeffs = tmp
    return coeffs[0]


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_convert_to_varsANF
def _convert_to_vars_anf(term: Sequence[int], variables: Sequence[Boolean]) -> Boolean:
    """The conjunction of the variables whose digit is 1 (``true`` for
    none)."""
    chosen = [variables[n] for n, t in enumerate(term) if t == 1]
    if not chosen:
        return true
    return and_evaluated(*chosen)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:bool_monomial
def bool_monomial(k: Union[int, Sequence[int]], variables: Sequence[Boolean]) -> Boolean:
    """The ``k``-th Zhegalkin monomial: the conjunction of the variables
    whose binary digit in ``k`` (an integer or the list of its digits,
    the first variable the most significant) is 1.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.anf import bool_monomial
    >>> x, y, z = symbols('x y z')
    >>> bool_monomial([1, 0, 1], [x, y, z]), bool_monomial(6, [x, y, z]), bool_monomial(0, [x, y, z])
    (x & z, x & y, True)
    """
    if isinstance(k, int):
        n = len(variables)
        if k < 0 or k >= 2**n:
            raise ValueError("Integer %d out of range for %d variables" % (k, n))
        term: list[int] = [(k >> (n - 1 - i)) & 1 for i in range(n)]
    else:
        term = list(k)
    return _convert_to_vars_anf(term, variables)


# PORTED FROM SYMPY: sympy/logic/boolalg.py:ANFform
def ANFform(variables: Sequence[Boolean], truthvalues: Sequence[int]) -> Boolean:
    """The algebraic normal form (Zhegalkin polynomial) of the function of
    ``variables`` whose truth table, in the order of
    ``itertools.product((0, 1), repeat=len(variables))``, is
    ``truthvalues``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.logic.anf import ANFform
    >>> x, y = symbols('x y')
    >>> ANFform([x], [1, 0]), ANFform([x, y], [0, 1, 1, 1]), ANFform([x, y], [1, 1, 1, 1])
    (x ^ True, x ^ y ^ (x & y), True)
    """
    n = len(variables)
    if len(truthvalues) != 2**n:
        raise ValueError("The number of truth values must be equal to 2^%d, got %d" % (n, len(truthvalues)))
    coeffs = anf_coeffs(truthvalues)
    terms = [t for i, t in enumerate(product((0, 1), repeat=n)) if coeffs[i] == 1]
    return xor_evaluated(*[_convert_to_vars_anf(t, variables) for t in terms], remove_true=False)
