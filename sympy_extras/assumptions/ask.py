"""Truth values of statements under assumptions."""
from __future__ import annotations

from sympy.assumptions import ask as _sympy_ask
from sympy.core.relational import Relational
from sympy.core.sympify import sympify
from sympy.logic.boolalg import (BooleanTrue, BooleanFalse, And, Or, Not,
    Implies, Equivalent, Xor, ITE, true, false)

from sympy_extras.polys.cad import truth_tables

from .context import global_assumptions
from .facts import Facts, normalize, to_polynomial, _predicate_of_atom
from .quantifiers import Quantifier

__all__ = ['ask']


def _facts(assumptions, domain, symbols):
    """The facts from the global assumptions and the explicit ones."""
    items = list(global_assumptions)
    if assumptions is not None:
        if isinstance(assumptions, (list, tuple, set, frozenset)):
            items.extend(assumptions)
        else:
            items.append(assumptions)
    return Facts(items, domain, symbols)


def _cad_ask(formula, facts):
    """Truth value of a Boolean combination of polynomial relations under
    the polynomial part of the facts, by cylindrical algebraic
    decomposition: ``True`` if it holds on every real point satisfying the
    assumptions, ``False`` if it fails on every such point, ``None``
    otherwise (or if the formula is not polynomial in real variables)."""
    poly = to_polynomial(formula, facts.real)
    if poly is None:
        return None
    if isinstance(poly, (BooleanTrue, BooleanFalse)):
        return bool(poly)
    premise = facts.polynomial
    gens = sorted(poly.free_symbols | premise.free_symbols, key=lambda s: s.name)
    if not gens:
        return bool(poly)
    _, (holds, values) = truth_tables([premise, poly], gens)
    relevant = [v for h, v in zip(holds, values) if h]
    if not relevant:
        # the assumptions have no real solution: nothing can be said
        return None
    if all(relevant):
        return True
    if not any(relevant):
        return False
    return None


def _evaluate_atom(atom, facts):
    if isinstance(atom, (BooleanTrue, BooleanFalse)):
        return bool(atom)
    predicate = _predicate_of_atom(atom)
    if predicate is not None:
        try:
            value = _sympy_ask(predicate, facts.predicates)
        except ValueError:
            value = None
        if value is not None:
            return value
    if isinstance(atom, Relational) and not atom.free_symbols:
        value = atom.doit()
        if value in (true, false):
            return bool(value)
    return _cad_ask(atom, facts)


def _evaluate(formula, facts):
    """Three-valued evaluation of a normalized formula: ``True``,
    ``False`` or ``None``."""
    if isinstance(formula, (BooleanTrue, BooleanFalse)):
        return bool(formula)
    if isinstance(formula, Not):
        value = _evaluate(formula.args[0], facts)
        return None if value is None else not value
    if isinstance(formula, (And, Or, Implies, Equivalent, Xor, ITE)):
        values = [_evaluate(arg, facts) for arg in formula.args]
        value = _combine(formula, values)
        if value is None:
            value = _cad_ask(formula, facts)
        return value
    return _evaluate_atom(formula, facts)


def _combine(formula, values):
    if isinstance(formula, And):
        if any(v is False for v in values):
            return False
        if all(v is True for v in values):
            return True
        return None
    if isinstance(formula, Or):
        if any(v is True for v in values):
            return True
        if all(v is False for v in values):
            return False
        return None
    if isinstance(formula, Implies):
        a, b = values
        if a is False or b is True:
            return True
        if a is True and b is False:
            return False
        return None
    if isinstance(formula, Equivalent):
        if any(v is None for v in values):
            return None
        return all(v == values[0] for v in values)
    if isinstance(formula, Xor):
        if any(v is None for v in values):
            return None
        return sum(values) % 2 == 1
    if isinstance(formula, ITE):
        c, a, b = values
        if c is True:
            return a
        if c is False:
            return b
        if a is not None and a == b:
            return a
        return None
    return None


def ask(query, assumptions=None, domain=None):
    """Truth value of a statement under assumptions.

    Parameters
    ==========

    query : Boolean
        A relation (``x > 0``, ``Eq(x, y)``), a membership
        (``element(x, S.Integers)``), a predicate (``Q.positive(x)``), a
        Boolean combination of those or a quantified formula
        (``ForAll(x, ...)``, ``Exists(x, ...)``).
    assumptions : Boolean or list of Booleans, optional
        Assumptions in the same form. The assumptions in
        :data:`~sympy_extras.assumptions.global_assumptions` are always
        used as well.
    domain : Set, optional
        A named SymPy set (``S.Reals``, ``S.Integers``, ...) all the
        variables are assumed to belong to.

    Returns
    =======

    ``True`` if the statement follows from the assumptions, ``False`` if
    its negation does and ``None`` if neither can be established.

    The assumptions and the query are first translated to the predicates
    of :mod:`sympy.assumptions` and answered by :func:`sympy.ask`; what
    remains undecided and is a Boolean combination of polynomial relations
    between real variables is decided exactly by cylindrical algebraic
    decomposition (see :mod:`sympy_extras.polys.cad`). Note that an
    inequality between expressions states that both sides are real.

    Examples
    ========

    >>> from sympy import S, Eq, Q
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions import ask, element, ForAll
    >>> ask(x**3 > 0, x > 0)
    True
    >>> ask(x**2 - 2*x + 1 >= 0, x > 0)
    True
    >>> ask(x > 1, x > 2)
    True
    >>> ask(x > 3, x > 2) is None
    True
    >>> ask(x*y > 0, (x > 0) & (y < 0))
    False
    >>> ask(element(x + 1, S.Integers), element(x, S.Integers))
    True
    >>> ask(x**2 >= 0) is None
    True
    >>> ask(x**2 >= 0, domain=S.Reals)
    True
    >>> ask(ForAll(x, x**2 + 2*x*y + y**2 >= 0), domain=S.Reals)
    True
    >>> ask(Q.positive(x), Eq(x, 3))
    True
    """
    query = sympify(query)
    if query is True or query is False:
        return bool(query)
    if query.has(Quantifier):
        from .resolve import resolve
        query = resolve(query, assumptions=assumptions, domain=domain)
    query = normalize(query)
    facts = _facts(assumptions, domain, query.free_symbols)
    return _evaluate(query, facts)
